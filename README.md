# IDR-RAG

Reproducibility package for **Information-Deficit Routing (IDR)**, a fixed-budget RAG allocator that estimates the value of the next prefix-legal retrieval action and reallocates a shared workload budget one action at a time.

This repository contains the paper-facing implementation and numeric release assets for the English V38 manuscript. It is a clean research release; it is not a copy of the surrounding MBA workspace.

## Reproduce the main table

The main replay uses frozen numeric action ledgers and makes no network, retrieval, or language-model calls.

```bash
python -m venv .venv
# activate .venv, then:
python -m pip install -r requirements-lock.txt
set PYTHONPATH=src
python scripts/reproduce_main.py
```

The audit should report `pass: true`, 11 policies, 3,600 actions, and `network_calls: 0`. The locked primary values include:

```text
IDR  AUBPC_0_50 = 55.72222484567901
DPA  AUBPC_0_50 = 48.24140663580247
IDR - DPA       =  7.480818209876539 percentage points
```

## Retrain the two released estimators

```bash
set PYTHONPATH=src
python scripts/train_models.py
```

This trains the 32-dimensional IDR regressor and Dynamic Benefit Probability classifier from the allocator-development ledger, then compares their predictions with the frozen evaluation scores. The expected maximum absolute errors are below `1e-10`.

## What is released

`artifacts/` contains hashed query identifiers, numeric states, F1 values, observed one-step changes, frozen policy scores, and recorded resource measurements. The query manifest records split provenance without author identity, question text, answers, reasoning, or model outputs. `artifacts/paper_results/` contains only the numeric result files used by the English V38 paper, including M-087 and M-088 outputs.

The original benchmark questions, gold answers, retrieved passage text, full trajectory JSONL files, API requests, provider metadata, and Elasticsearch indexes are intentionally not redistributed. They remain subject to their own sources and licenses. Reconstructing those upstream inputs is optional and is not needed for the exact paper replay supplied here.

## Repository layout

```text
src/idr_rag/          feature extraction, models, allocation, and metrics
scripts/              replay, retraining, verification, and bootstrap utilities
tools/                maintainer-only numeric artifact exporter
artifacts/            public numeric ledgers and paper result files
models/               retrained release estimators
tests/                tests of the paper's allocation invariants
docs/                 data, reproducibility, and third-party notices
```

The code is licensed under Apache-2.0. Benchmark content and any upstream model or software components remain under their original terms; see `docs/THIRD_PARTY_LICENSES.md`.
