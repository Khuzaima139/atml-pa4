# ATML PA4: Efficient Language Models

Code for Programming Assignment 4 of EE-5102/CS-6304 Advanced Topics in Machine Learning (Fall 2026).

| Task | Topic |
| --- | --- |
| Task 1 | Model compression: pruning and quantization of Qwen2.5-1.5B |
| Task 2 | Exact and efficient attention, associative recall, and KV-cache compression |
| Task 3 | Draft-model distillation for speculative decoding |

## Setup (Mac)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/env_check.py
.venv/bin/python -m pytest -q
```

## Data

```bash
.venv/bin/python scripts/prepare_data.py
```

It clones the course data repository at a pinned commit, runs its public preparation with the source revisions in `data_lock/local_source_revisions.json`, and checks content hashes against `data_lock/public_assets.json`.

## Dependency pins

| File | Use |
| --- | --- |
| `requirements.txt` | Mac environment (Python 3.11): direct requirements, including `lm_eval==0.4.13` and the packages its installation added |
| `requirements-kaggle.txt` | Kaggle: installed with `pip install -r requirements-kaggle.txt -c constraints.txt`, where `constraints.txt` pins the preinstalled torch, which stays unchanged |
| `requirements-lock.txt` | Full `pip freeze` of the Mac environment |

Hugging Face downloads use `HF_HUB_DISABLE_XET=1`.

## Task 1: pruning and quantization of Qwen2.5-1.5B

Model: `Qwen/Qwen2.5-1.5B` at the revision locked in `task1/config.py`, FP16. Conditions (`task1/config.py`): FP16; P1 magnitude and P2 Wanda pruning at 25/50/75%; P3 Wanda with a block-level OWL-style allocation at 50/75%; 2:4 Wanda at 50%; Q1 round-to-nearest and Q2 AWQ-style group-wise asymmetric quantization (group 128) at 8/4/3 bits; Q3 mixed precision at the Q2 INT4 and INT3 packed-byte budgets.

### Data

`scripts/prepare_data.py` (above) produces the locked files used by Task 1: 128 C4 calibration sequences and 32 C4 development sequences of 512 tokens, WikiText-2 test and C4 validation windows (50,000 scored tokens each) and 2,000 HellaSwag validation IDs. Calibration, probes and allocation selection read only the calibration and development files; the held-out files are refused inside these steps (`task1.data.forbid_held_out`).

### Runs, in order

Kaggle runs use two T4 GPUs. Each Kaggle notebook clones this repository at a pushed commit into `/kaggle/working/atml-pa4`, records the preinstalled torch and installs `requirements-kaggle.txt` with it pinned, runs `python scripts/prepare_data.py` and `python scripts/env_check.py`, runs a GPU preflight and a timing smoke of the real code paths, writes the job lists with `python -m task1.kaggle_runN jobs`, runs one `python -m common.kaggle_queue` process per GPU, then `summary`, `check` and `zip` of the same module. The zip is imported on the Mac into `results/task1/` (sha256-checked, `IMPORT_MANIFEST.json` per run).

1. **Kaggle, run 1** (`python -m task1.kaggle_run1 jobs --stage run1 --jobs-dir <dir>`): calibration statistics and AWQ scale search, block sensitivity probes, development-set selection of the P3 and block-level Q3 allocations, and every condition built and measured on calibration and development data:

   ```bash
   python -m task1.calib_stats --out results/task1/calib_stats
   python -m task1.probe --kind prune --calib-stats results/task1/calib_stats --out results/task1/probe_prune
   python -m task1.probe --kind quant --calib-stats results/task1/calib_stats --out results/task1/probe_quant
   python -m task1.select --kind p3 --target s50 --probe results/task1/probe_prune --calib-stats results/task1/calib_stats --out results/task1/select_p3_s50
   python -m task1.select --kind q3 --target int4 --probe results/task1/probe_quant --calib-stats results/task1/calib_stats --out results/task1/select_q3_int4
   python -m task1.compress --condition p2_s50 --artifacts /kaggle/working/artifacts/p2_s50 --out results/task1/run1/p2_s50
   python -m task1.compress --condition q2_b4 --artifacts /kaggle/working/artifacts/q2_b4 --calib-stats results/task1/calib_stats --out results/task1/run1/q2_b4
   python -m task1.compress --condition p3_s50 --artifacts /kaggle/working/artifacts/p3_s50 --allocation results/task1/select_p3_s50/selected_allocation.json --out results/task1/run1/p3_s50
   ```

   (likewise for every target and condition; the full lists are written by the `jobs` command). The block-level Q3 selections collapsed to the uniform allocation and are kept as an unsuccessful design.
2. **Kaggle, run 1b** (`python -m task1.kaggle_run1b jobs --stage run1b --jobs-dir <dir>`): Q3 with fold-group units (`task1.probe --kind quant_v2 --blocks 0-13` and `14-27` into `results/task1/probe_quant_v2/`, `task1.select --kind q3v2` into `results/task1/select_q3v2_int4` and `select_q3v2_int3`, `task1.compress` of `q3_int4` and `q3_int3` into `results/task1/run1b/`). The artifacts of runs 1 and 1b are kept as notebook outputs.
3. **Mac, freeze**: the selected P3 and Q3 allocations are committed as `task1/frozen/*.json` before any held-out evaluation; the evaluation refuses a frozen file that is uncommitted or modified.
4. **Kaggle, run 2** (`python -m task1.kaggle_run2 locate`, then `jobs --stage run2`), with the run-1 and run-1b notebook outputs attached: artifacts verified against their committed manifests, then held-out evaluation and, after all evaluations, resource measurement on one GPU:

   ```bash
   python -m task1.baseline --no-resources --out results/task1/run2/fp16
   python -m task1.evaluate_condition --condition p3_s50 --artifacts <run-1 artifacts>/p3_s50 --run1 results/task1/run1/p3_s50 --no-resources --out results/task1/run2/p3_s50
   python -m task1.baseline --resources-only --out results/task1/run2_resources/fp16
   python -m task1.evaluate_condition --condition p3_s50 --artifacts <run-1 artifacts>/p3_s50 --run1 results/task1/run1/p3_s50 --resources-only --out results/task1/run2_resources/p3_s50
   ```

   (`--run1 results/task1/run1b/<ID>` for `q3_int4` and `q3_int3`).
5. **Mac, analysis**: rebuilds every Task 1 table, figure and qualitative case from the committed results; the output directory must not exist:

   ```bash
   .venv/bin/python -m task1.analysis --out results/task1/analysis
   ```

On the Mac, `.venv/bin/python -m pytest -q` runs every pipeline piece end to end with tiny random models and tiny data.

### Results

| Path | Content |
| --- | --- |
| `results/task1/calib_stats/` | Dense calibration statistics, AWQ scales and search record |
| `results/task1/probe_prune/`, `probe_quant/`, `probe_quant_v2/` | Block and fold-group sensitivities with probe effort |
| `results/task1/select_*/` | Candidate allocations with development and calibration NLL, selected allocation |
| `results/task1/run1/`, `run1b/` | Per condition: build record, per-projection output error and score distributions, calibration and development NLL per sequence, artifact manifest |
| `results/task1/run2/` | Per condition: perplexity per window (both suites), HellaSwag scores per example and ending, development NLL reproduction; FP16 lm_eval comparison |
| `results/task1/run2_resources/` | Per condition: storage accounting, prefill and decode latency, peak allocated memory |
| `results/task1/analysis/` | Tables T1-T6 and diagnostics (CSV, `summary.json`), figures F1-F5 (PDF), `qualitative.jsonl` and `qualitative.txt` |
| `results/task1/figures/` | report figures (F1-F5, F3_full), rebuilt with `.venv/bin/python -m task1.analysis --figures-only --out results/task1/figures` |

## Task 2: exact attention, associative recall and KV-cache compression

Protocol, predictions and dated amendments (AM1-AM4): `task2/PROTOCOL.md`, fixed before the results they govern.

### Data

| Source | Pin | sha256 |
| --- | --- | --- |
| Course data repository | commit `6a0a9ec9da7ef3d313ee3036cfec493886723644` (`data_lock/public_assets.json`) | |
| MQAR generator `scripts/mqar_generator.py` (course supplied, T 256) | data repository commit above | `af69fe1cfd03a63ddbe2cc5844691bff138f06206bd94e8c7cb7aad93c727d08` |
| MQAR development set `data/mqar/mqar_development.jsonl.gz` | data repository commit above | `b960a5c503956afdb77edb43c9c89003ca5cbed07cca3ca65fbc71d50be5d9fd` |
| MQAR test set `data/mqar/mqar_test.jsonl.gz` | data repository commit above | `84925dc5f2b7a7e0b042fc354e5ec34435002b829a90b35813accd005ad4b549` |
| Retrieval cases `data/kv_retrieval/kv_retrieval_cases.jsonl.gz` (180) | data repository commit above | `6e2c12dae84805ed54666ff647a06e51d9412adde65729635ce07a0caa64fc13` (decompressed `e12679b6a07e03cb6330b4a5987bd149953e46907dd9a1ebaa4936cdfca2b1d9`) |
| `Qwen/Qwen2.5-1.5B-Instruct` | revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16, SDPA | |

File hashes are checked against the data repository's `SHA256SUMS.json` before use and recorded in every run.json. MQAR training sequences are generated with the course generator for each (K, regime, seed) stream and checked for disjointness against `task2/heldout_hashes.json.gz` (written by `python -m task2.make_heldout_hashes`).

### Runs, in order

Kaggle notebooks use two T4 GPUs and follow the Task 1 pattern: clone this repository at a pushed commit, pin the preinstalled torch, install `requirements-kaggle.txt`, run `python scripts/prepare_data.py` and `python scripts/env_check.py`, a preflight and smoke run of the real entry points, a plan, one queue per GPU, then `summary`, `check` and `zip` of the notebook module. Each zip is imported on the Mac (sha256-checked, never overwriting) with a manifest in `results/task2/`.

1. **Mac**: `python -m task2.make_heldout_hashes` (held-out hash list, committed before training).
2. **Kaggle, notebook `atml-pa4-t2-nb1`** (code commit `c99b65d`; `python -m task2.kaggle_nb1 jobs --stage main --jobs-dir <dir>`): Part A and the B1/B2 training runs and sweep.

   ```bash
   python -m task2.part_a verify --out results/task2/part_a/verify
   python -m task2.part_a bench --out results/task2/part_a/bench
   python -m task2.train_mqar --mechanism dense --K 16 --regime mixed --seed 6304 --out results/task2/mqar/dense_mixed_K16_s6304
   python -m task2.mqar_sweep --out results/task2/mqar_sweep
   ```

   (`train_mqar` for 13 configurations x seeds 6304, 6305, 6306; the full list is written by the `jobs` command). Manifest `results/task2/IMPORT_MANIFEST.json`.
3. **Kaggle, notebook `atml-pa4-t2-nb2`** (code commit `1376175`; `python -m task2.kaggle_nb2 jobs --stage main --jobs-dir <dir>`): the B3 curriculum runs (AM3) and Part C.

   ```bash
   python -m task2.train_mqar --mechanism local_dilated --K 16 --regime mixed --curriculum near_to_mixed --seed 6304 --out results/task2/mqar/local_dilated_curriculum_K16_s6304
   python -m task2.kv_eval --mode validate --out results/task2/part_c/validate
   python -m task2.kv_eval --mode eval --out results/task2/part_c/eval
   python -m task2.kv_bench --out results/task2/part_c/bench
   ```

   Manifest `results/task2/IMPORT_MANIFEST_nb2.json`.
4. **Kaggle, notebook `atml-pa4-t2-nb3`** (code commit `c3e3d06`; `python -m task2.kaggle_nb3 jobs --stage main --jobs-dir <dir>`): the post hoc AM4 diagnostic of the token-wise INT4 result.

   ```bash
   python -m task2.kv_diag --out results/task2/part_c/diag
   ```

   Manifest `results/task2/IMPORT_MANIFEST_nb3.json`.
5. **Mac, analysis**: every Task 2 table, figure, prediction outcome and the qualitative export, rebuilt from the committed results (headline numbers recomputed from the per-example files and checked against each run.json); the output directory must not exist:

   ```bash
   .venv/bin/python -m task2.analysis --out results/task2/analysis
   .venv/bin/python -m task2.analysis --figures-only --out results/task2/figures
   ```

The 42 final MQAR checkpoints (`model_final.safetensors`) are not committed; they are preserved in the outputs of the notebook versions above, with sha256 in `IMPORT_MANIFEST.json` and `IMPORT_MANIFEST_nb2.json` and in each run's `run.json` (`checkpoint_sha256`).

### Results

| Path | Content |
| --- | --- |
| `results/task2/part_a/verify/`, `bench/` | Output and gradient errors per implementation, dtype and L; forward and forward-plus-backward timing and peak memory; SDPA backend table; score-matrix bytes |
| `results/task2/mqar/<mechanism>_<regime>_K<K>_s<seed>/` | Training log, development curve, per-query development and test predictions (`*.npz`), run record |
| `results/task2/mqar_sweep/` | Forward-only sweep at T 256-8192 |
| `results/task2/part_c/validate/`, `eval/`, `bench/`, `diag/` | Cache-pipeline validation; per-prompt rows of every method and per-prompt position arrays (`npz/`); qualitative text files; packed-cache benchmark; AM4 diagnostic |
| `results/task2/analysis/` | Tables A (`A_*.csv`), B1-B3 (`B*.csv`), C1-C5 (`C*.csv`), prediction outcomes `P_predictions.csv`, `C_qualitative.txt` and `.jsonl`, `summary.json`, figures A1-A2, B1-B3, C1-C2 (PDF); rebuilt with `.venv/bin/python -m task2.analysis --out results/task2/analysis` |
| `results/task2/figures/` | Report figures, rebuilt with `.venv/bin/python -m task2.analysis --figures-only --out results/task2/figures` |

## Attribution

- Course data repository [AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets) (commit `6a0a9ec`), whose scripts prepare the fixed dataset selections and are called rather than copied. Its MQAR generator (`scripts/mqar_generator.py`) produces the Task 2 training streams; it is loaded from the data repository after a sha256 check, not copied.
- Models from Hugging Face: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Datasets: [allenai/c4](https://huggingface.co/datasets/allenai/c4), [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag), [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).
- lm-evaluation-harness 0.4.13: its HellaSwag preprocessing function is imported for prompt formatting, and its HFLM log-likelihood is used to check the scorer on the FP16 model.
