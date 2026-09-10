# Reproducibility contract

The release has three levels:

1. **Exact replay.** `scripts/reproduce_main.py` replays all eleven policies from numeric evaluation actions and checks every reported checkpoint and AUBPC value against `artifacts/expected_results.json`.
2. **Allocator retraining.** `scripts/train_models.py` refits the IDR and Dynamic Benefit Probability estimators from the development ledger and verifies their predictions against the frozen scores.
3. **Upstream regeneration.** Recreating benchmark retrieval and reader trajectories requires the original benchmark sources and provider configuration. It is outside the exact replay contract and is not required to reproduce the paper's allocation results.

All allocation orders must satisfy strict prefix legality: a depth-2 action can enter the eligible set only after its depth-1 action has been selected. The evaluated budget is an exact quota over the released action pool.
