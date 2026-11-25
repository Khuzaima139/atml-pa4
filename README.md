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

## Attribution

- Course data repository [AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets) (commit `6a0a9ec`), whose scripts prepare the fixed dataset selections and are called rather than copied.
- Models from Hugging Face: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Datasets: [allenai/c4](https://huggingface.co/datasets/allenai/c4), [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag), [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).
