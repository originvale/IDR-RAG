# Data and artifact scope

The public ledgers are derived from two frozen 1,800-query banks: an allocator-development set and an independent evaluation set, with 600 queries per benchmark. They contain hashed or opaque query identifiers, split and dataset labels, numeric 32-dimensional state features, F1 values, one-step F1 changes, frozen comparator scores, and recorded resource values.

The release does not include question text, gold answers, candidate answers, reasoning traces, passage text, API metadata, or the benchmark indexes. This avoids redistributing third-party benchmark content and keeps the release free of provider information. The manifest and the artifact checksums document the derivation of the public files.

The exact paper replay consumes `artifacts/evaluation_actions_32d.csv`; it does not need to regenerate retrieval trajectories. The development ledger is used by `scripts/train_models.py` for retraining.
