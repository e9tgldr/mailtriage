"""MailTriage CLI -- sort a real mailbox, or rehearse on the offline corpus.

Safe by default: without --apply nothing is moved; the tool prints exactly what it would do.
It never deletes mail -- `move` copies first and only then removes the original.

Examples
--------
  # rehearse against the labeled corpus, no account needed
  python run.py --corpus

  # see what would happen to the 20 newest real messages (still moves nothing)
  python run.py --user me@gmail.com --limit 20

  # actually file them under MailTriage/<folder>
  python run.py --user me@gmail.com --limit 20 --apply

Password: pass --password / MAILTRIAGE_PASSWORD (an APP PASSWORD, never the account password).
This script stores nothing; the value lives in memory for the process only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify import TIER_RULES, classify  # noqa: E402
from rules import FOLDERS, REVIEW  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "inbox.json")


def build_scorer(disabled: bool):
    if disabled:
        return None, "disabled (--rules-only)"
    try:
        from ml import LayaScorer

        scorer = LayaScorer()
        if scorer.load():
            return scorer, scorer.note
        return None, "unavailable: %s" % scorer.note
    except Exception as exc:  # noqa: BLE001
        return None, "unavailable: %s" % exc


def target_folder(decision, prefix: str, review_folder: str | None):
    """Map a decision to a destination. Review is a real folder only when asked for."""
    if decision.folder == REVIEW:
        if review_folder is None:
            return None
        return "%s/%s" % (prefix, review_folder) if prefix else review_folder
    return "%s/%s" % (prefix, decision.folder) if prefix else decision.folder


def show(message_id: str, subject: str, decision, dest: str | None) -> None:
    where = dest or "(left in place - held for review)"
    print(
        "%-28s -> %-14s conf=%.2f %-6s | %s"
        % (message_id[:28], decision.folder, decision.confidence, decision.tier, where)
    )
    print("    subj: %s" % (subject[:90] or "(none)"))
    for reason in decision.reasons[:3]:
        print("    why : %s" % reason)
    if decision.caveat:
        print("    note: %s" % decision.caveat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", action="store_true", help="rehearse on corpus/inbox.json (no account)")
    parser.add_argument("--user", default=os.environ.get("MAILTRIAGE_USER", ""))
    parser.add_argument("--password", default=os.environ.get("MAILTRIAGE_PASSWORD", ""))
    parser.add_argument("--host", default=os.environ.get("MAILTRIAGE_HOST", "imap.gmail.com"))
    parser.add_argument("--mailbox", default="INBOX")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prefix", default="MailTriage", help="destination parent folder")
    parser.add_argument("--review-folder", default="Review",
                        help="folder for held messages; empty string leaves them in place")
    parser.add_argument("--rules-only", action="store_true", help="skip the ML tier entirely")
    parser.add_argument("--apply", action="store_true", help="actually move messages (default: rehearse)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    scorer, ml_note = build_scorer(args.rules_only)

    # ------------------------------------------------------------------ corpus rehearsal
    if args.corpus:
        with open(CORPUS, encoding="utf-8") as handle:
            corpus = json.load(handle)
        records = []
        for msg in corpus:
            decision = classify(msg["subject"], msg["body"], msg.get("headers", ""), scorer=scorer)
            dest = target_folder(decision, args.prefix, args.review_folder or None)
            records.append(
                {
                    "id": msg["id"],
                    "subject": msg["subject"],
                    "gold": msg["gold"],
                    "dest": dest,
                    "correct": decision.folder == msg["gold"],
                    **decision.as_dict(),
                }
            )
        if args.json:
            print(json.dumps({"ml": ml_note, "records": records}, ensure_ascii=False, indent=2))
            return 0
        print("ML tier: %s\n" % ml_note)
        for rec in records:
            mark = "ok  " if rec["correct"] else ("MISS" if rec["folder"] != REVIEW else "held")
            print("%-8s %-16s -> %-16s %s" % (rec["id"], rec["gold"], rec["folder"], mark))
        auto = [r for r in records if r["folder"] != REVIEW]
        good = sum(int(r["correct"]) for r in auto)
        print("-" * 72)
        print("auto-filed %d, all correct: %s" % (len(auto), good == len(auto)))
        print("held for review: %d" % sum(1 for r in records if r["folder"] == REVIEW))
        return 0

    # ------------------------------------------------------------------ live mailbox
    if not args.user or not args.password:
        parser.error("need --user and --password (or MAILTRIAGE_USER / MAILTRIAGE_PASSWORD)")

    from imap_client import Inbox

    review_root = args.review_folder or None
    plan: list[tuple[str, str, object, str | None]] = []
    with Inbox(args.user, args.password, args.host, args.mailbox) as inbox:
        messages = inbox.newest(args.limit)
        print(
            "%s: %d message(s) from %s | ML tier: %s | mode: %s\n"
            % (args.host, len(messages), args.mailbox, ml_note, "APPLY" if args.apply else "rehearsal")
        )
        for mail in messages:
            decision = classify(mail.subject, mail.body, mail.headers, scorer=scorer)
            dest = target_folder(decision, args.prefix, review_root)
            plan.append((mail.uid, mail.subject, decision, dest))
            show(mail.uid, mail.subject, decision, dest)

        counts = {f: 0 for f in FOLDERS + [REVIEW]}
        for _, _, decision, _ in plan:
            counts[decision.folder] = counts.get(decision.folder, 0) + 1
        print("\nplan: %s" % ", ".join("%s=%d" % (k, v) for k, v in counts.items() if v))

        if not args.apply:
            print("\nrehearsal only - nothing was moved. re-run with --apply to file them.")
            return 0

        moved = failed = 0
        for uid, subject, decision, dest in plan:
            if dest is None:
                continue
            try:
                inbox.move(uid, dest)
                moved += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("  FAILED %s -> %s: %s" % (uid, dest, exc))
        print("\nmoved %d, failed %d, held %d"
              % (moved, failed, sum(1 for _, _, _, d in plan if d is None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
