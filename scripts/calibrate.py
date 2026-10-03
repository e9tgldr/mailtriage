"""Calibration + honesty harness.

Runs the labeled corpus through the cascade and prints what the product would actually do:
accuracy BY TIER, the quarantine rate, and the confusion between folders.

The point of this file is to make the claims checkable. Any number quoted in the README
must come from here. Usage (must use the project venv so torch/laya are importable):

    C:\\Users\\hi\\Documents\\laya-mn\\venv\\Scripts\\python.exe scripts\\calibrate.py
    ... scripts\\calibrate.py --no-ml      # rules tier only (no GPU, no Laya)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules import FOLDERS, REVIEW  # noqa: E402
from classify import TIER_GATE, TIER_ML, TIER_RULES, classify  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "inbox.json")
ALL_LABELS = FOLDERS + [REVIEW]


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ml", action="store_true", help="rules tier only")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    scorer = None
    ml_note = "disabled (--no-ml)"
    if not args.no_ml:
        try:
            from ml import LayaScorer

            scorer = LayaScorer()
            if not scorer.load():
                ml_note = "unavailable: %s" % scorer.note
                scorer = None
            else:
                ml_note = scorer.note
        except Exception as exc:  # noqa: BLE001
            ml_note = "import failed: %s" % exc
            scorer = None

    print("=" * 96)
    print("corpus      : %s (%d messages)" % (args.corpus, len(corpus)))
    print("ML tier     : %s" % ml_note)
    print("=" * 96)

    results = []
    correct = Counter()
    total_by_gold = Counter()
    confuse = defaultdict(Counter)

    for msg in corpus:
        decision = classify(msg["subject"], msg["body"], msg.get("headers", ""), scorer=scorer)
        gold = msg["gold"]
        total_by_gold[gold] += 1
        hit = decision.folder == gold
        correct[gold] += int(hit)
        confuse[gold][decision.folder] += 1
        results.append((msg, decision, hit))
        flag = "ok  " if hit else "MISS"
        print(
            "%-8s %-16s -> %-16s %s conf=%.2f tier=%-5s"
            % (msg["id"], gold, decision.folder, flag, decision.confidence, decision.tier)
        )
        if decision.reasons:
            print("         why: %s" % "; ".join(decision.reasons[:3]))

    # ---- aggregate
    n = len(corpus)
    n_ok = sum(correct.values())
    print("-" * 96)
    print("OVERALL top-1: %d/%d = %.3f" % (n_ok, n, n_ok / n))
    by_tier = Counter(d.tier for _, d, _ in results)
    print(
        "decided by tier: rules=%d  ml=%d  gated->review=%d"
        % (by_tier[TIER_RULES], by_tier[TIER_ML], by_tier[TIER_GATE])
    )

    print("\nPer-folder accuracy (gold='review' means it SHOULD be held back):")
    for gold in ALL_LABELS:
        if not total_by_gold[gold]:
            continue
        print("  %-16s %d/%d" % (gold, correct[gold], total_by_gold[gold]))

    print("\nAuto-filed subset only (messages the product did NOT hold for review):")
    auto = [(m, d, h) for m, d, h in results if d.folder != REVIEW]
    if auto:
        a_ok = sum(int(h) for _, _, h in auto)
        print("  %d/%d = %.3f correct" % (a_ok, len(auto), a_ok / len(auto)))
        wrong = [(m["id"], m["gold"], d.folder, d.tier) for m, d, h in auto if not h]
        if wrong:
            print("  mis-filed (the failures that matter most):")
            for mid, gold, got, tier in wrong:
                print("    %s: %s -> %s (tier %s)" % (mid, gold, got, tier))
        else:
            print("  mis-filed: 0  <- nothing was filed into the wrong folder")
    else:
        print("  nothing auto-filed")

    held = [(m["id"], m["gold"]) for m, d, _ in results if d.folder == REVIEW]
    print("\nHeld for review: %d/%d (%.0f%%)" % (len(held), n, 100 * len(held) / n))
    for mid, gold in held:
        note = "" if gold == REVIEW else "  <- gold was %s (correct call: unsure, not wrong)" % gold
        print("  %s%s" % (mid, note))

    print("\nConfusion (gold -> predicted):")
    for gold in ALL_LABELS:
        if confuse[gold]:
            print("  %-16s %s" % (gold, dict(confuse[gold])))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "n": n,
                    "accuracy": n_ok / n,
                    "by_tier": dict(by_tier),
                    "auto_filed": len(auto),
                    "auto_filed_correct": sum(int(h) for _, _, h in auto) if auto else 0,
                    "per_folder": {
                        g: {"correct": correct[g], "n": total_by_gold[g]}
                        for g in ALL_LABELS
                        if total_by_gold[g]
                    },
                },
                handle,
                indent=2,
            )
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
