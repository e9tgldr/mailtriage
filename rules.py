"""Tier 1 of the cascade: deterministic routing rules for Mongolian inboxes.

Why a rules tier exists at all (evidence, not preference): the Laya-MN checkpoint is
overconfident on out-of-distribution email prose -- in the 2026-09-22 probe it routed a
government notice to vendor_spam at p=0.98. So the ML model is NOT allowed to decide.
Tier 1 owns every case it can decide with evidence it can show the user.

Design rules for this module:
  * stdlib only, no imports outside the standard library
  * every hit is returned as a human-readable reason -- the UI shows WHY, not just where
  * patterns are matched case-insensitively against subject (weighted x2) + body
  * Cyrillic and Latin spellings both listed: Mongolian senders mix them freely
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Folders. Order matters: it is the tie-break order when scores are equal.
# --------------------------------------------------------------------------------------
FOLDERS = [
    "invoice_payment",
    "school",
    "government",
    "vendor_spam",
    "personal",
]

# A message the rules cannot decide goes to the ML tier; if that is also unsure it lands
# in review. This folder is never a model output -- it is the safety net.
REVIEW = "review"

# --------------------------------------------------------------------------------------
# Rules. Each entry: folder -> (compiled pattern, weight, human label)
# weight 3 = decisive vocabulary (a Mongolian would never use it in another context)
# weight 2 = strong signal
# weight 1 = weak / supporting signal
# --------------------------------------------------------------------------------------
_RAW_RULES: dict[str, list[tuple[str, int, str]]] = {
    "invoice_payment": [
        (r"нэхэмжлэх|нэхэмжлэл", 3, "invoice"),
        (r"и-баримт|е-баримт|ebarimt|e-barimt", 3, "e-barimt"),
        (r"төлбөр|төлөх|төлнө|шилжүүлнэ|шилжүүлэх", 2, "payment"),
        (r"дансны дугаар|iban|данс", 2, "bank account"),
        (r"төгрөг|₮|mnt\b", 2, "amount in MNT"),
        (r"\d[\d,\.]{3,}\s*(төгрөг|₮|mnt)", 3, "explicit amount"),
        (r"нөхөн төлбөр|торгууль|үлдэгдэл", 2, "balance/penalty"),
    ],
    "school": [
        (r"хичээл|шалгалт|дүн|анги|сурагч|багш", 2, "school vocabulary"),
        (r"эцэг эх|эцэг\s*эхийн", 3, "parent meeting"),
        (r"хурал|уулзалт", 1, "meeting"),
        (r"сургууль|сургалт|боловсрол", 2, "school/education"),
        (r"гэрийн даалгавар|даалгавар", 2, "homework"),
        (r"дүнгийн хуудас|табель|журнал", 3, "grade record"),
        (r"багшийн|ангийн удирдлага", 2, "teacher/class admin"),
        (r"@school\.|\.edu\.mn", 3, "school domain"),
    ],
    "government": [
        (r"хүсэлт|хүсэлтийг|өргөдөл", 2, "application/request"),
        (r"албан ёс|албан бичиг|албан тоот", 3, "official document"),
        (r"хороо|яам|газрын|агентлаг|захиргаа", 3, "authority"),
        (r"дүүрэг|нийслэл|төрийн", 2, "district/state"),
        (r"хууль|тушаал|шийдвэр|зөвшөөрөл|лиценз|гэрчилгээ", 2, "law/licence"),
        (r"давж гаргах|хянан үзэ", 2, "processing language"),
        (r"@gov\.mn|\.gov\.mn", 3, "gov domain"),
        (r"иргэний|татварын|төрийн алба", 2, "civic"),
    ],
    "vendor_spam": [
        (r"хямдрал|хямдруул|зөвхөн өнөөдөр|яаралтай", 3, "urgency/discount"),
        (r"бэлэг|урамшуулал|бонус", 2, "gift/promo"),
        (r"худалдан авах|захиалга өгөх|дарахад л", 2, "call to buy"),
        (r"хүргэлт үнэгүй|урамшуулалтай", 2, "delivery promo"),
        (r"дарж|товшиж|линк|link", 1, "click"),
        (r"зар|сурталчилгаа|маркетинг", 2, "ad"),
        (r"%\s*(хямдрал|off)", 3, "percent-off"),
        (r"subscribe|unsubscribe|newsletter", 2, "mailing list"),
    ],
    "personal": [
        (r"сайн байна уу|сайн уу|баярлалаа|баярлаа", 1, "greeting"),
        (r"уулзах уу|уулзъя|утсаар|дууд", 2, "meet/call"),
        (r"гэртээ|гэр бүл|аав|ээж|ах|эгч|дүү", 2, "family"),
        (r"төрсөн өдөр|баяр хүргэе", 2, "personal occasion"),
        (r"чи|та хоёр|найз", 1, "informal address"),
    ],
}

RULES: dict[str, list[tuple[re.Pattern[str], int, str]]] = {
    folder: [(re.compile(p, re.IGNORECASE), w, lbl) for p, w, lbl in entries]
    for folder, entries in _RAW_RULES.items()
}

# Strong signals that a message is machine-generated bulk mail regardless of topic.
BULK_HEADER_SIGNALS = [
    (r"^<(bulk|junk|spam)>", 3, "bulk/junk mail header"),
    (r"list-unsubscribe", 3, "list-unsubscribe header"),
    (r"precedence:\s*(bulk|junk)", 2, "bulk precedence"),
    (r"no-?reply|do-?not-?reply", 2, "no-reply sender"),
]

# --------------------------------------------------------------------------------------
# Decision thresholds. Calibrated by scripts/calibrate.py on the labeled corpus -- see
# docs/CALIBRATION.md. Do not hand-tune these; re-run the calibration instead.
# --------------------------------------------------------------------------------------
RULE_NET = 4        # net lead of the winning folder over the runner-up
RULE_TOTAL = 5      # and at least this much total evidence
ML_MIN = 0.80       # ML tier may decide only at/above this probability...
ML_LEAD = 0.25      # ...and only with this much lead over the next folder

# --- guards added 2026-09-22 after the first full-cascade calibration exposed two failures.
# gov-03 ("Татварын урьдчилсан тооцоо") was routed to invoice_payment at p=0.99 on a money-word
# shortcut, and amb-03 ("Re: төслийн явц", 5 words) was routed to vendor_spam at p=0.98.
# Both are the overconfidence signature from the probe -- high probability, no real evidence.

# 1. When the rules tier already has evidence, ML must beat it decisively before overriding it.
ML_OVERRIDE_MIN = 0.95   # probability needed to overrule a rule tier that has any signal
ML_RULE_CONFLICT_MIN = 0.70  # and the ML's own 2nd choice must own at least this much of the
                             # mass before a rule-vs-ML conflict is considered real. gov-03
                             # ran 0.99 vs 0.01 -- a one-horse race on a single word is noise.

# 2. Short messages carry too little text for a 192-token model to be trustworthy. A five-word
# message can only be classified by its author's intent, not by vocabulary.
ML_MIN_BODY_CHARS = 60



@dataclass
class RuleResult:
    """What tier 1 concluded. `decided=False` hands the message to the ML tier."""
    scores: dict[str, int]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    decided: bool = False
    folder: str | None = None

    @property
    def ranked(self) -> list[tuple[str, int]]:
        return sorted(self.scores.items(), key=lambda kv: (-kv[1], FOLDERS.index(kv[0])))

    def top(self) -> tuple[str, int]:
        return self.ranked[0]


def _window(subject: str, body: str) -> tuple[str, str]:
    return subject or "", body or ""


def score_message(subject: str, body: str, raw_headers: str = "") -> RuleResult:
    """Tier 1. Subject hits count double; headers are searched for the bulk signals."""
    subject, body = _window(subject, body)
    scores: dict[str, int] = {f: 0 for f in FOLDERS}
    reasons: dict[str, list[str]] = {f: [] for f in FOLDERS}

    for folder, entries in RULES.items():
        for pattern, weight, label in entries:
            hit_subject = bool(pattern.search(subject))
            hit_body = bool(pattern.search(body))
            if hit_subject or hit_body:
                pts = weight * (2 if hit_subject else 1)
                scores[folder] += pts
                where = "subject" if hit_subject else "body"
                reasons[folder].append(f"{label} ({where}, +{pts})")

    header = (raw_headers or "").lower()
    for pattern, weight, label in BULK_HEADER_SIGNALS:
        if re.search(pattern, header, re.IGNORECASE | re.MULTILINE):
            scores["vendor_spam"] += weight
            reasons["vendor_spam"].append(f"{label} (+{weight})")

    res = RuleResult(scores=scores, reasons=reasons)
    ranked = res.ranked
    if len(ranked) >= 2:
        top_folder, top_score = ranked[0]
        runner_up = ranked[1][1]
        net = top_score - runner_up
        if top_score >= RULE_TOTAL and net >= RULE_NET:
            res.decided = True
            res.folder = top_folder
    return res
