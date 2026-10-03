"""The cascade: rules -> ML tie-breaker -> confidence gate.

This ordering is the whole design, and it is not negotiable. It exists because the ML model
is wrong at p=0.98 on email prose (measured -- docs/CALIBRATION.md). The gate is what keeps
that failure away from the user's mail: an unsure message goes to `review` instead of being
filed wrongly, because a wrongly-filed email is worse than an unfiled one.

Three guards protect the gate, each added because a concrete message defeated it (2026-09-22):

  G1. `rules_has_evidence` -- if the rules tier found ANY signal, ML may only override the
      rules' own leader when it is both very confident (ML_OVERRIDE_MIN) and MI-3 disagreed
      in a way that is not a one-horse race. Without this, gov-03 (a tax notice with the word
      "тооцоо") was filed to invoice_payment at p=0.99, beating a correct government=6 rule.
  G2. `ML_RULE_CONFLICT_MIN` -- an ML distribution of 0.99 vs 0.01 on out-of-distribution
      text is the model fixing on one vocabulary token, not reading the message.
  G3. `ML_MIN_BODY_CHARS` -- a 5-word message ("Re: төслийн явц") was filed to vendor_spam at
      p=0.98. Short text cannot support a confident route; hold it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# Run this file directly (python classify.py) or import it as a module -- both must work,
# so the sibling modules are resolved by inserting their own directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import (  # noqa: E402
    FOLDERS,
    ML_LEAD,
    ML_MIN,
    ML_MIN_BODY_CHARS,
    ML_OVERRIDE_MIN,
    ML_RULE_CONFLICT_MIN,
    REVIEW,
    score_message,
)

TIER_RULES = "rules"
TIER_ML = "ml"
TIER_GATE = "gate"


@dataclass
class Decision:
    folder: str
    confidence: float
    tier: str
    reasons: list[str] = field(default_factory=list)
    rules_scores: dict[str, int] = field(default_factory=dict)
    ml_probs: dict[str, float] | None = None
    caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "folder": self.folder,
            "confidence": round(self.confidence, 4),
            "tier": self.tier,
            "reasons": self.reasons,
            "rules_scores": self.rules_scores,
            "ml_probs": (
                {k: round(v, 4) for k, v in self.ml_probs.items()} if self.ml_probs else None
            ),
            "caveat": self.caveat,
        }


def _rules_leader(scores: dict[str, int]) -> tuple[str | None, int]:
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], FOLDERS.index(kv[0])))
    folder, points = ranked[0]
    return (folder if points > 0 else None), points


def classify(subject: str, body: str, headers: str = "", scorer=None) -> Decision:
    """Route one message. `scorer` is an optional ml.LayaScorer (lazy, may be unavailable)."""
    rules = score_message(subject, body, headers)

    # ---- tier 1: rules decided it outright
    if rules.decided and rules.folder:
        folder, pts = rules.top()
        # Confidence is the *lead*, not a probability -- the rules tier proves evidence,
        # it does not estimate probability. Keep that distinction visible in the number.
        lead = pts - (rules.ranked[1][1] if len(rules.ranked) > 1 else 0)
        return Decision(
            folder=folder,
            confidence=min(1.0, lead / max(pts, 1)),
            tier=TIER_RULES,
            reasons=rules.reasons[folder],
            rules_scores=rules.scores,
        )

    # ---- G3: too little text for a confident route, whatever the model says
    body_chars = len((body or "").strip())
    too_short = body_chars < ML_MIN_BODY_CHARS

    # ---- tier 2: ML tie-breaker on the ambiguous remainder
    if scorer is not None:
        result = scorer.score(f"{subject}\n{body}".strip(), FOLDERS)
        if result.available and result.probs:
            folder, prob = result.top()
            lead = prob - result.runner_up()
            rule_folder, rule_points = _rules_leader(rules.scores)
            conflicts_with_rules = rule_folder is not None and rule_folder != folder

            # G1 + G2: overriding a rules signal needs a much higher bar, and the model's own
            # runner-up must carry real mass (a 0.99/0.01 split is one-token fixation).
            override_ok = (not conflicts_with_rules) or (
                prob >= ML_OVERRIDE_MIN and result.runner_up() >= ML_RULE_CONFLICT_MIN
            )
            decisive = prob >= ML_MIN and lead >= ML_LEAD and override_ok and not too_short

            if decisive:
                return Decision(
                    folder=folder,
                    confidence=prob,
                    tier=TIER_ML,
                    reasons=[
                        "rules inconclusive (%s)"
                        % (", ".join("%s=%d" % (f, s) for f, s in rules.ranked if s) or "no hits"),
                        "ML tie-break %.2f vs %.2f" % (prob, result.runner_up()),
                    ]
                    + (
                        ["overrode rules leader %s (%d pts)" % (rule_folder, rule_points)]
                        if conflicts_with_rules
                        else []
                    ),
                    rules_scores=rules.scores,
                    ml_probs=result.probs,
                    caveat="ML-ranked: probabilities are not calibrated on email.",
                )

            # ---- tier 3: gate. Unsure stays unfiled.
            why = ["ML was not decisive (%.2f vs %.2f)" % (prob, result.runner_up())]
            if too_short:
                why.append(
                    "message too short to route confidently (%d chars < %d)"
                    % (body_chars, ML_MIN_BODY_CHARS)
                )
            if conflicts_with_rules and not override_ok:
                why.append(
                    "ML disagreed with rules leader %s (%d pts) without meeting override bar %.2f"
                    % (rule_folder, rule_points, ML_OVERRIDE_MIN)
                )
            return Decision(
                folder=REVIEW,
                confidence=prob,
                tier=TIER_GATE,
                reasons=why,
                rules_scores=rules.scores,
                ml_probs=result.probs,
                caveat="Held for human review rather than risk mis-filing.",
            )
        if not result.available:
            return Decision(
                folder=REVIEW,
                confidence=0.0,
                tier=TIER_GATE,
                reasons=["rules inconclusive; ML unavailable: %s" % (result.note or "unknown")],
                rules_scores=rules.scores,
            )

    # ---- no ML at all: rules-only deployment still has to be honest
    return Decision(
        folder=REVIEW,
        confidence=0.0,
        tier=TIER_GATE,
        reasons=[
            "rules inconclusive (%s)"
            % (", ".join("%s=%d" % (f, s) for f, s in rules.ranked if s) or "no rule hits")
        ]
        + (["message too short (%d chars)" % body_chars] if too_short else []),
        rules_scores=rules.scores,
    )
