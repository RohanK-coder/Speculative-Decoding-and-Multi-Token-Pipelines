# Speculative Decoding and Multi-Token Pipelines

A software prototype and architecture study of **speculative decoding** for large language model inference, built for **CECS 530 / Advanced Computer Architecture**.

This project treats speculative decoding as a **pipeline, control, and memory-system design problem**, not only as a text-generation trick. It studies draft generation, target verification, token commit, rollback, backpressure, and KV-cache management as interacting stages of a multi-token inference pipeline.

---

## Reproduction Note

This repository provides two Git branches for different evaluation needs:

- **`demo-48-run-sweep`**: reduced configuration for quick grading, demo walkthroughs, and reproducibility checks.
- **`main`**: full configuration for broader experimental analysis.

Recommended workflow:

- Use **`demo-48-run-sweep`** for quick verification.
- Use **`main`** for extended experiments and report-level analysis.

---

## Quick Start

Use this path first when checking the project locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=. python experiments/validate_correctness.py
PYTHONPATH=. python experiments/run_single.py
PYTHONPATH=. pytest -q
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH="."
python experiments\validate_correctness.py
python experiments\run_single.py
pytest -q
```

---



## 1. Project Summary

In baseline autoregressive decoding, the target model generates **one token at a time**. This creates a sequential dependency chain where latency scales directly with output length and target-model cost.

In speculative decoding, a **smaller draft model** proposes a block of future tokens, and a **larger target model** verifies that block. The system then:

1. commits the **longest accepted prefix**,
2. rolls back any rejected speculative suffix,
3. appends a corrective target token when needed,
4. updates the decoding state and KV-cache model,
5. and repeats until generation finishes.

This repository studies that process from three perspectives:

- **Algorithmic correctness**: only verified tokens are committed.
- **Pipeline architecture**: draft, verify, commit, rollback, and fallback behave like a multi-stage pipeline.
- **Systems tradeoffs**: speedup depends on acceptance rate, draft/target cost ratio, speculation depth, verification cost, rollback overhead, and KV-cache behavior.

Key questions explored:

- How can speculative decoding reduce the one-token-at-a-time bottleneck of autoregressive generation?
- When does speculation outperform baseline decoding?
- How do acceptance rate, speculation depth, rollback, verification cost, backpressure, and KV-cache overhead affect speed?
- Can adaptive or hybrid control policies make speculative decoding more robust across prompts and model families?

---

## 2. What Is Original in This Project?

This project is inspired by published speculative decoding work, but the implementation and evaluation framework are project-specific.

Project-specific contributions include:

- a clean baseline greedy decoder for comparison,
- a configurable speculative decoder with speculation depth `k`,
- rollback-safe state handling,
- KV-cache consistency modeling,
- adaptive speculation-depth control,
- hybrid-gated fallback behavior,
- category-aware policy settings,
- architecture-style metrics for backpressure, stalls, wasted draft work, and KV-cache overhead proxies,
- experiment scripts for single runs, comparisons, grid sweeps, and plot generation,
- a Streamlit dashboard for interactive inspection.

The strongest technical idea in this project is the **adaptive multi-token speculative pipeline**: the system does not treat speculation depth as a fixed constant only, but studies how control policy changes affect acceptance, rollback, and speedup.

---

## 3. Repository Layout

```text
Speculative-Decoding-and-Multi-Token-Pipelines/
├── core/
│   ├── analysis.py
│   ├── baseline.py
│   ├── cache.py
│   ├── config.py
│   ├── metrics.py
│   ├── models.py
│   ├── naive_multitoken.py
│   ├── prompting.py
│   ├── speculative.py
│   ├── streaming_decode.py
│   └── utils.py
├── experiments/
│   ├── compare_algorithms.py
│   ├── compare_model_families.py
│   ├── export_best_configs.py
│   ├── plot_results.py
│   ├── run_grid.py
│   ├── run_single.py
│   └── validate_correctness.py
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── results/
├── tests/
├── app.py
├── README.md
└── requirements.txt
```

> A local `.venv/` directory may exist on the developer machine, but it should be ignored by Git and recreated locally by each user.

---

## 4. Supported Model Families

The project is designed around small draft/target configurations so that basic workflows can run on local machines.

```python
MODEL_FAMILIES = {
    "gpt2": {
        "draft": "distilgpt2",
        "target": "gpt2",
        "label": "GPT-2 family",
        "recommended_k": 3,
        "recommended_tokens": 32,
        "recommended_mode": "hybrid",
    },
    "tinyllama": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama (LLaMA-family)",
        "recommended_k": 2,
        "recommended_tokens": 48,
        "recommended_mode": "hybrid",
    },
    "qwen": {
        "draft": "Qwen/Qwen2.5-1.5B-Instruct",
        "target": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5-1.5B-Instruct",
        "recommended_k": 2,
        "recommended_tokens": 48,
        "recommended_mode": "hybrid",
    },
    "smollm2": {
        "draft": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "target": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 Draft/Target",
        "recommended_k": 3,
        "recommended_tokens": 32,
        "recommended_mode": "hybrid",
    },
}
```

On limited hardware, start with **`distilgpt2 -> gpt2`**.

Some larger or gated models may require more memory, Hugging Face authentication, or license approval.

---

## 5. Environment Setup

### Recommended Python Version

Use **Python 3.10-3.12**.

Python 3.13 may work for some parts, but ML package compatibility can be inconsistent.

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify the environment

```bash
python -c "import torch, transformers, streamlit, matplotlib, pandas; print('env ok')"
```

### Notes

- Run commands from the project root.
- Use `PYTHONPATH=.` when running scripts from the command line.
- Use the `gpt2` family first on CPU-only machines.
- For CUDA-specific PyTorch wheels, install PyTorch using the official PyTorch selector before installing the rest of the packages.

---

## 6. Reproducibility Guide

### Correctness check

```bash
PYTHONPATH=. python experiments/validate_correctness.py
```

### Single run

```bash
PYTHONPATH=. python experiments/run_single.py
```

### Streamlit app

```bash
PYTHONPATH=. streamlit run app.py
```

Recommended first app configuration:

```text
family: gpt2
mode: hybrid
k: 2 or 3
max new tokens: 32
device: cpu
```

### Model-family comparison

```bash
PYTHONPATH=. python experiments/compare_model_families.py
```

### Algorithm comparison

```bash
PYTHONPATH=. python experiments/compare_algorithms.py
```

### Grid sweep

```bash
PYTHONPATH=. python experiments/run_grid.py
```

### Plot generation

```bash
PYTHONPATH=. python experiments/plot_results.py
```

---

## 7. Testing

Run all tests:

```bash
PYTHONPATH=. pytest -q
```

Recommended test coverage:

- cache commit, rollback, discard, and checkpoint behavior,
- metric aggregation and derived ratios,
- speculative accept-all behavior,
- speculative rollback and corrective-token behavior,
- hybrid fallback behavior after low acceptance,
- configuration defaults.

The lightweight mock-based tests should run without downloading Hugging Face models. The model-equivalence test may download `distilgpt2` and `gpt2`, so it can be kept as an integration test.

---

## 8. Expected Outputs

Experiment outputs should be written under:

```text
outputs/results/
outputs/plots/
outputs/logs/
```

Recommended result files:

```text
outputs/results/grid_results.csv
outputs/results/grid_summary.csv
outputs/results/category_summary.csv
outputs/results/grid_results.json
```

Recommended plot files:

```text
outputs/plots/speedup_vs_k.png
outputs/plots/acceptance_vs_k.png
outputs/plots/speedup_vs_acceptance.png
outputs/plots/speedup_vs_cost_ratio.png
outputs/plots/energy_per_token_proxy.png
outputs/plots/slowdown_combined_realistic.png
```

If these files are included in the repository, the README should show one small results table and link to the generated plots.

---

## 9. Results Summary

Replace the sample values below with your actual measured results from `outputs/results/`.

| Setup | Mode | k | Max Tokens | Acceptance Rate | Speedup | Rollbacks | Wasted Draft Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 -> gpt2 | fixed | 2 | 32 | TBD | TBD | TBD | TBD |
| distilgpt2 -> gpt2 | adaptive | 3 | 32 | TBD | TBD | TBD | TBD |
| distilgpt2 -> gpt2 | hybrid | 3 | 32 | TBD | TBD | TBD | TBD |

Suggested figures to include in the report and README:

1. **Speedup vs speculation depth `k`**
2. **Acceptance rate vs `k`**
3. **Speedup vs acceptance rate**
4. **Speedup vs draft/target cost ratio**
5. **Energy-per-token proxy vs `k`**
6. **Rollback count or wasted draft tokens vs `k`**

---

## 10. Expected Findings

The expected conclusions are:

- speculative decoding is **not always faster** than baseline,
- acceptance rate is the most important performance indicator,
- speculation depth `k` has an optimal middle range,
- very large `k` can cause diminishing returns because rollback and verification overhead increase,
- a cheap draft model is important for useful speedup,
- adaptive and hybrid policies can improve robustness,
- speedup alone is not enough; memory overhead and energy proxy should also be considered.

In one sentence:

> Speculative decoding outperforms baseline only when accepted speculative work is large enough to amortize verification, rollback, and control overhead.

---

## 11. Limitations

- Some larger model families require more memory than a typical laptop provides.
- Gated models may require Hugging Face authentication and accepted license terms.
- CPU-only runs are supported but may be slower for larger sweeps.
- Energy results are proxy-based unless collected from actual device instrumentation.
- The full grid experiment can be time-consuming on CPU-only systems.
- Results depend heavily on model pair, prompt category, and speculation depth.

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'core'`

Run scripts from the project root with `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

### Hugging Face model download or authentication issues

Start with `distilgpt2` and `gpt2`. If using gated models, confirm Hugging Face login and accepted terms.

### Plot script cannot find result files

Run the grid experiment first:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

Then regenerate plots:

```bash
PYTHONPATH=. python experiments/plot_results.py
```

---

## 13. Reproducibility Checklist

- [ ] Python 3.10-3.12 used
- [ ] virtual environment created
- [ ] pinned dependencies installed from `requirements.txt`
- [ ] correctness check passes
- [ ] `pytest -q` passes
- [ ] single-run experiment completes
- [ ] grid results are exported
- [ ] plots are regenerated
- [ ] README results table is updated with real values
- [ ] sample CSV and plot files are committed if required by the course

---

## 14. Future Work

Possible extensions include:

- cycle-accurate architectural simulation,
- hardware-assisted rollback,
- explicit queue and backpressure modeling,
- real power and energy measurement,
- broader GPU-backed benchmark sweeps,
- larger prompt sets,
- stronger draft-target model hierarchies,
- comparison against production inference runtimes.

---

## 15. References / Starting Points

Useful references and systems:

- Fast Inference from Transformers via Speculative Decoding
- Accelerating Large Language Model Decoding with Speculative Sampling
- Blockwise Parallel Decoding for Deep Autoregressive Models
- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- Better & Faster Large Language Models via Multi-token Prediction
- Hugging Face Transformers
- vLLM
- llama.cpp
- PyTorch

---

## 16. Authors

- Yanni Rohan Kommathoti
- Nikhil Peravali
- Rajiv Sai Charan Tirumalasetti
