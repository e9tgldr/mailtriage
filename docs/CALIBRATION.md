# Calibration log — MailTriage

Every number quoted in the README comes from `scripts/calibrate.py`. This file records what
changed and what it did, so a future session can tell measurement from guesswork.

Command:

```
C:\Users\hi\Documents\laya-mn\venv\Scripts\python.exe scripts\calibrate.py --json-out reports\calibration.json
```

Corpus: `corpus/inbox.json`, 18 labeled messages (5 hard cases, 6 probe messages verbatim,
3 adversarial-short messages, 4 genuinely ambiguous). ML: Laya-MN epoch 8, loaded on CUDA
(missing=0, unexpected=0).

## Run 1 — 2026-09-22, pre-guard

| Metric | Value |
|---|---|
| Overall top-1 | 16/18 = 0.889 |
| Decided by | rules=10, ml=6, gated=2 |
| **Auto-filed subset** | **14/16 = 0.875 — 2 MIS-FILED** |
| Mis-filed | `gov-03` government→invoice_payment (ml, p=0.99); `amb-03` review→vendor_spam (ml, p=0.98) |
| Held for review | 2/18 = 11% |

Read this as the failure that justified the guards: overall accuracy *looked* better than
Run 2 while the product was actually worse, because two messages were filed into the wrong
folder at high confidence. **Overall top-1 is the wrong metric** — the metric that matters is
"of the messages we actually filed, how many were right".

The two failures had the same shape: high probability, no evidence. `gov-03` had a correct
rule signal (`government=6`) that the ML overrode on the money word "тооцоо"; `amb-03` was
5 words long with zero rule hits and was routed anyway.

## Run 2 — 2026-09-22, post-guard (current)

Guards added to `rules.py`: `ML_OVERRIDE_MIN=0.95`, `ML_RULE_CONFLICT_MIN=0.70`,
`ML_MIN_BODY_CHARS=60`. Logic in `classify.py`.

| Metric | Run 1 | **Run 2** |
|---|---|---|
| Overall top-1 | 16/18 | **17/18 = 0.944** |
| Decided by | rules10/ml6/gate2 | rules10/**ml4**/**gate4** |
| Auto-filed | 14 | 14 |
| **Auto-filed correct** | 14/16 = 0.875 | **14/14 = 1.000** |
| **Mis-filed** | **2** | **0** |
| Held for review | 2 (11%) | 4 (22%) |

Both Run 1 failures are now held for review with an explicit reason instead of being mis-filed.
The cost is honest: quarantine went 11% → 22% because `amb-03` and `gov-03` moved from
"confidently wrong" to "held". That trade is correct — an unfiled message costs a click, a
mis-filed invoice costs trust in the product.

## Rules-only (no GPU, `--rules-only`)

| Metric | Value |
|---|---|
| Auto-filed | 10 |
| Auto-filed correct | **10/10 = 1.000** |
| Held for review | 8/18 = 44% |

The ML tier's job, quantified: it rescued 4 messages (sch-02, sch-03, spam-03, per-02) from
the review pile while adding **zero** mis-filings. That is exactly the intended division of
labour, and it is the argument for keeping the 643 MB checkpoint in the loop.

## Next measurement (not yet done)

- Real-world number: needs ~300 labeled real messages. Nothing here substitutes for that.
- The corpus is self-authored and the rules were fitted to it — this is plumbing evidence.
- `amb-01` ("Хурал / 17:00") sits at ML 0.32 vs 0.29. Watch whether real ambiguous mail
  clusters there; if so, add a bulk-undo path before shipping.
