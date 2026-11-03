# Execution records

`latest.json` points only to the completed, verified three-policy comparison.

| Run (UTC) | Status | Scope |
|---|---|---|
| `run_20261103T020926Z` | Complete; all checks passed | MI, EMSE, and normalized 50/50 fusion on the same Intel replay. |
| `run_20261103T020824Z` | Failed setup attempt; retained for diagnosis | Analytic and numerical checks passed, but the first MI worker stopped on a duplicate logger timestamp keyword. No completed policy comparison is reported from this attempt. |

Runs are never overwritten. `.cache/` holds disposable reporting caches and is excluded from Git. Dense target maps are GP predictions, not measured ground truth. The parent README and executed Notebook summarize only the verified run.
