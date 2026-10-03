# MailTriage — Mongolian inbox routing

Working engine for the email-sorting product. The product name is not decided yet (see
**Name** below); this directory is deliberately named after the function, not the brand, so
renaming is a display-layer change and nothing else.

## What actually works right now

| Claim | Evidence |
|---|---|
| Routes Mongolian mail into 5 folders | `scripts/calibrate.py` on `corpus/inbox.json` |
| **0 messages filed into the wrong folder** | 14 auto-filed, 14 correct |
| 22% held for human review instead of guessed | 4 of 18 held |
| Runs with **zero API cost** | local Laya-MN checkpoint, local GPU, no network calls |
| Works with no GPU at all | `--rules-only` still auto-files 10/10 correct |

Re-run it yourself — every number above comes from this command:

```
C:\Users\hi\Documents\laya-mn\venv\Scripts\python.exe scripts\calibrate.py
```

## Architecture (the order is the design)

```
message ──▶ TIER 1 rules (rules.py)        deterministic, explainable, subject weighted x2
               │ decided? ──────────────▶ file it, showing the reasons
               ▼ inconclusive
            TIER 2 Laya-MN (ml.py)         643 MB local checkpoint, ranks 5 folders
               │ confident? ────────────▶ file it (guarded, see below)
               ▼
            TIER 3 gate (classify.py)      → review folder. NEVER guess.
```

**Why ML is not allowed to decide.** Measured 2026-09-22: Laya-MN classified real Mongolian
email prose into arbitrary inbox folders at 3/6 (~3x chance) — but it was wrong *at 0.98
confidence* (a government notice read as vendor spam). A model that is confidently wrong
cannot own the decision. So the rules tier decides what it can, ML only breaks ties, and the
gate keeps the rest unfiled. A wrongly-filed email is worse than an unfiled one.

**Three guards**, each added because a specific message defeated the gate (all in
`rules.py`, all documented there):

| Guard | The message that forced it |
|---|---|
| `ML_OVERRIDE_MIN = 0.95` — ML must clear a much higher bar to overrule a rules signal | `gov-03` tax notice → `invoice_payment` at p=0.99, beating a correct `government=6` |
| `ML_RULE_CONFLICT_MIN = 0.70` — a 0.99-vs-0.01 split is one-token fixation, not reading | same message; a one-horse race on the word "тооцоо" |
| `ML_MIN_BODY_CHARS = 60` — short text cannot support a confident route | `amb-03` "Re: төслийн явц" (5 words) → `vendor_spam` at p=0.98 |

## Usage

```bash
# rehearse on the labeled corpus — no account, no network
python run.py --corpus

# rehearse on your 20 newest real messages — still moves nothing
python run.py --user you@gmail.com --limit 20

# actually file them under MailTriage/<folder>
python run.py --user you@gmail.com --limit 20 --apply

# no GPU / no Laya install: rules tier alone
python run.py --rules-only --corpus

# machine-readable
python run.py --corpus --json
```

Password is an **app password** (Gmail: Account → Security → 2-Step Verification → App
passwords), passed via `--password` or `MAILTRIAGE_PASSWORD`. Nothing is written to disk.

**Safety properties baked in:** dry-run by default; `move` copies first and only removes the
original after the copy succeeds; no delete path exists anywhere in this code.

## Files

| File | Role |
|---|---|
| `rules.py` | Tier 1 rules, folder list, all thresholds |
| `ml.py` | Tier 2 — lazy Laya adapter (reuses `laya-mn/out_mn`, no 643 MB copy) |
| `classify.py` | The cascade + the three guards |
| `imap_client.py` | Stdlib IMAP: fetch, create folder, safe move |
| `run.py` | CLI (corpus rehearsal + live mailbox, dry-run default) |
| `corpus/inbox.json` | 18 labeled messages, including all 6 probe messages verbatim |
| `scripts/calibrate.py` | Honesty harness — accuracy by tier, quarantine rate, confusion |
| `reports/calibration.json` | Latest measured result |

Thresholds in `rules.py` are **calibrated, not hand-tuned**. Re-run `scripts/calibrate.py`
after changing any of them, and put the new numbers in `docs/CALIBRATION.md`.

## Name

`laya-mn` is the model, not the product — and it will be dropped entirely once the engine is
confident, so the name must not depend on it. Candidates, with the one fact I could verify:

| Name | Meaning | Status |
|---|---|---|
| **Shuud** (шууд) | "directly / straight" — message goes straight where it belongs | `shuud.mn` **registered**; `shuud.com` taken; `shuud.app` appears free |
| **Shuudan** (шуудан) | "mail / post" — the Mongolian word for postal mail | `shuudan.mn` no A record → possibly available |
| **Kemel** (кэмэл) | "rhythm / measure" — sorting as keeping time | `kemel.mn` no A record → possibly available |

Domain checks via DNS + RDAP on 2026-09-22; `.mn` names without an A record are *hints*, not
proof — confirm at a registrar before printing anything.

## Known limits — read before promising anything

- The 18-message corpus was **written by me, and the rules were written against it.** That is
  self-confirming by construction. It proves the pipeline runs and that the gate holds; it is
  NOT a measurement of real-world accuracy.
- To get a real number you need ~300 labeled real messages. Until then, treat the accuracy
  figures as plumbing evidence.
- `review` held 22% of the self-written corpus, and real mail is messier — expect more.
- The ML tier's probabilities are **not calibrated on email**. Never show them to a user as a
  confidence percentage without relabeling.
- Folder set is fixed at 5 + review. Per-user custom folders are not supported yet.
