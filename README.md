# Speculative Decoding and Multi-Token Pipelines

A software prototype and architecture study of **speculative decoding** for large language model inference, built for **CECS 530 / Advanced Computer Architecture**.

> **Reproduction note**
>
> This project provides two Git branches for different evaluation needs:
>
> - **`demo-48-run-sweep`**: a reduced configuration with **48 checks** that completes in **less than 2 minutes** on a typical local machine and still generates outputs and plots.
> - **`main`**: the full configuration with **960 checks** that takes about **20 minutes** on a typical local machine and is intended for more complete evaluation and it generates outputs and plots.
>
> For quick verification, use **`demo-48-run-sweep`**. For extended experiments, use **`main`**.


This project studies speculative decoding as more than a text-generation trick. It treats the problem as a **pipeline, control, and memory-system design problem**:

- How do we break the one-token-at-a-time barrier of autoregressive decoding?
- When does speculation actually outperform baseline decoding?
- What do **acceptance rate, speculation depth, verification cost, rollback, backpressure, and KV-cache management** do to end-to-end speed?
- How can **adaptive** and **hybrid** control policies make speculative decoding more robust?

---

## 1. Project Summary

In baseline autoregressive decoding, the target model generates **one token at a time**, so latency scales with output length and target-model cost.

In speculative decoding, a **smaller draft model** proposes a block of future tokens and a **larger target model** verifies that block. The system then:

1. commits the **longest accepted prefix**,
2. rolls back the rejected suffix,
3. appends a corrective target token if needed,
4. and repeats until generation finishes.

This repository studies that process from three perspectives:

- **Algorithmic correctness** — only verified tokens are committed.
- **Pipeline architecture** — draft, verify, commit, and rollback stages interact like a multi-token pipeline.
- **Systems tradeoffs** — speedup depends on acceptance rate, draft/target cost ratio, speculative KV-cache overhead, and control-policy quality.

---

## 2. Main Contributions

This project includes:

- a **baseline greedy decoder** for reference,
- a **speculative decoder** with configurable speculation depth `k`,
- **rollback-safe state handling**,
- **KV-cache consistency modeling**,
- multiple control policies:
  - fixed,
  - adaptive,
  - hybrid-gated,
  - category-aware,
- a **Streamlit web app** for interactive runs,
- experiment scripts and plot generation code,
- analytical and empirical views of:
  - speedup,
  - acceptance rate,
  - verification bottlenecks,
  - stall rounds,
  - backpressure,
  - wasted draft work,
  - energy/memory tradeoffs.

---

## 3. Repository Layout

This README matches the **current root-level project structure**:

```text
Speculative-Decoding-and-Multi-Token-Pipelines/
├── .venv/
├── core/
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
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

### Important files

- `app.py`  
  Streamlit interface for baseline vs speculative decoding, live metrics, and interactive demos.

- `core/`  
  Core model wrappers, baseline decoding, speculative decoding, analysis helpers, and utilities.

- `experiments/run_single.py`  
  Run one experiment configuration for quick testing.

- `experiments/validate_correctness.py`  
  Validate that speculative decoding only commits verified tokens and handles rollback correctly.

- `experiments/run_grid.py`  
  Batch experiment runner for parameter sweeps.

- `experiments/compare_model_families.py`  
  Compare multiple model families under similar decoding settings.

- `experiments/compare_algorithms.py`  
  Compare baseline and speculative approaches.

- `experiments/plot_results.py`  
  Generate plots from exported CSV results.

- `outputs/results/`  
  CSV and JSON experiment outputs.

- `outputs/plots/`  
  Generated figures for the report or slides.

- `outputs/logs/`  
  Optional run logs.

---

## 4. Supported Model Families

The project was designed around small draft/target configurations such as:

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
    "llama32": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target": "meta-llama/Llama-3.2-3B-Instruct",
        "label": "TinyLlama -> Llama-3.2-3B-Instruct",
        "recommended_k": 2,
        "recommended_tokens": 48,
        "recommended_mode": "hybrid",
    },
}
```

On limited hardware, start with **`distilgpt2 -> gpt2`**.

---

## 5. Environment Setup

### Recommended Python version

Use **Python 3.10–3.12**.

Python 3.13 may work for some parts, but ML package support and `torch` / `torchvision` compatibility can be inconsistent.

### Create and activate a virtual environment

Create a local virtual environment so dependencies are **not installed globally**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify the environment

```bash
python -c "import torch, transformers, streamlit, matplotlib, pandas; print('env ok')"
```

### Notes

- Run all commands from the **project root**, the folder that contains `app.py`, `core/`, `experiments/`, and `requirements.txt`.
- Activate `.venv` before running the app or experiments.
- Some Hugging Face models may require authentication or license approval.
- `meta-llama/Llama-3.2-3B-Instruct` may require a Hugging Face login and accepted terms.
- On limited hardware, use smaller families first.

---

## 6. Reproducability Guide

This section gives one clear workflow for launching the app, running experiments, exporting results, and generating plots.
After Running Experimental Setup (Part 5) do the following steps.

### Step 1 — Activate the virtual environment

Before running anything:

```bash
source .venv/bin/activate
```

### Step 2 — Run the app

Launch the Streamlit dashboard from the project root:

```bash
PYTHONPATH=. streamlit run app.py
```

The app is the easiest way to confirm the project is working.

Inside the app, you can:

- choose a model family,
- compare baseline decoding and speculative decoding,
- vary speculation depth `k`,
- select policies such as `fixed`, `adaptive`, or `hybrid`,
- observe metrics such as acceptance rate, rollback behavior, and speedup.

### Recommended first app run

Use a lightweight setup first:

- family: `gpt2`
- mode: `hybrid`
- `k = 2` or `k = 3`
- max new tokens: `32`
- device: `cpu`

This is the safest first run for correctness and basic performance checks.



### Step 3 — Run one experiment configuration

For a quick test that does not require a large sweep:

```bash
PYTHONPATH=. python experiments/run_single.py
```

Use this when you want one reproducible example for a demo or grading walkthrough.

### Step 4 — Run a comparison script

You can also run a more targeted comparison:

```bash
PYTHONPATH=. python experiments/compare_model_families.py
```

or:

```bash
PYTHONPATH=. python experiments/compare_algorithms.py
```

These are useful for focused comparisons without running the full grid.

### Step 5 — Run the grid experiment

The full grid runner is:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

This generates structured outputs for multiple combinations of:

- prompts,
- speculation depth `k`,
- output length,
- policy mode,
- and speed-ratio cases.

**Important:** the default `run_grid.py` may run many combinations. For a submission or professor demo, use a reduced configuration if you want faster reproduction.

### Step 6 — Check exported results

Experiment outputs should appear under:

```text
outputs/results/
outputs/plots/
outputs/logs/
```

Typical result files include:

```text
outputs/results/grid_results.csv
outputs/results/grid_summary.csv
outputs/results/category_summary.csv
outputs/results/grid_results.json
```

### Step 7 — Generate plots

After results are exported, generate figures with:

```bash
PYTHONPATH=. python experiments/plot_results.py
```

If your plotting script accepts explicit arguments, use the exported CSV in `outputs/results/` and write figures into `outputs/plots/`.

Typical plot outputs include:

```text
outputs/plots/speedup_vs_k.png
outputs/plots/acceptance_vs_k.png
outputs/plots/speedup_vs_acceptance.png
outputs/plots/speedup_vs_cost_ratio.png
outputs/plots/energy_per_token_proxy.png
outputs/plots/slowdown_combined_realistic.png
```

### Step 8 — Recommended end-to-end workflow

For the cleanest reproduction path:

1. create and activate `.venv`,
2. install dependencies,
3. launch the app with `PYTHONPATH=. streamlit run app.py`,
4. run `validate_correctness.py`,
5. run `run_single.py` or a comparison script,
6. run `run_grid.py` if you want a larger sweep,
7. run `plot_results.py` to generate figures from exported results.

---

## 7. Fast Reproduction Path for Grading or Demo

A professor or TA usually does **not** need to run a full exhaustive sweep.

Recommended quick path:

```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run app.py
```

Then, in a second terminal:

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/validate_correctness.py
PYTHONPATH=. python experiments/run_single.py
PYTHONPATH=. python experiments/plot_results.py
```

This path is much more practical than a very large grid search and is usually enough to demonstrate:

- the project runs correctly,
- baseline and speculative decoding are implemented,
- speculative metrics are emitted,
- outputs and plots can be generated.

If you keep a larger sweep for your own analysis, describe it as an **extended evaluation mode**, not the default grading path.

---

## 8. Expected Outputs and Plot Set

The most useful outputs and figures for this project are:

### CSV / JSON outputs

Recommended logged fields include:

- prompt
- category
- max new tokens
- mode
- `k`
- acceptance rate
- speedup
- rollback count
- baseline fallback steps
- wasted draft tokens
- draft time
- verify time
- commit time
- rollback penalty
- pipeline utilization
- stall rounds
- backpressure events
- verify bottleneck ratio
- energy proxy
- KV-cache overhead proxy
- output match

### Recommended plots

1. **Speedup vs speculation depth `k`**  
   Shows latency tradeoffs and diminishing returns.

2. **Acceptance rate vs `k`**  
   Shows why speedup rises or falls.

3. **Speedup vs acceptance rate**  
   Shows the strongest causal relationship behind speculation efficiency.

4. **Speedup vs draft/target cost ratio**  
   Shows why a cheap draft model matters.

5. **Energy-per-token proxy vs `k`**  
   Shows that the fastest setting is not always the most efficient one.

6. **Slowdown analysis plots**  
   Useful for diagnosing:
   - verify bottleneck,
   - backpressure,
   - stall rounds,
   - wasted draft tokens.

---

## 9. Suggested Commands

### App

```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run app.py
```

### Correctness check

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/validate_correctness.py
```

### Single run

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/run_single.py
```

### Model-family comparison

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/compare_model_families.py
```

### Algorithm comparison

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/compare_algorithms.py
```

### Full grid

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/run_grid.py
```

### Plot generation

```bash
source .venv/bin/activate
PYTHONPATH=. python experiments/plot_results.py
```

---

## 10. Expected Findings

Across the project, the main expected conclusions are:

- speculative decoding is **not always faster** than baseline,
- **acceptance rate** is the most important performance indicator,
- `k` has an **optimal middle range**,
- very large `k` causes diminishing returns because rollback and verification overhead rise,
- a **cheap draft model** is important,
- adaptive and hybrid control policies improve robustness,
- speedup alone is not enough; memory overhead and energy proxy also matter.

In one sentence:

> Speculative decoding outperforms baseline only when accepted work is large enough to amortize verification, rollback, and control overhead.

---

## 11. Limitations

- Some larger model families require more memory than a typical laptop provides.
- Gated models may require authentication.
- A subset of presentation figures may be trend illustrations rather than large-scale measured sweeps.
- Energy results are proxy-based unless collected from actual device instrumentation.
- The full grid experiment can be time-consuming on CPU-only systems.

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'core'`

Run scripts from the project root with `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

### `python3.11: command not found`

Use the Python version you have installed:

```bash
python3 --version
python3 -m venv .venv
```

### `.venv/bin/activate: no such file or directory`

The virtual environment was not created successfully yet. First run:

```bash
python3 -m venv .venv
```

### `python: command not found`

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

### Hugging Face model download or auth issues

Start with `distilgpt2` and `gpt2`. If you later use gated models, make sure your Hugging Face login and accepted terms are configured.

---

## 13. Reproducibility Statement

This repository is designed to be **functionally reproducible**:

- the decoding logic can be rerun,
- the experiments can be reswept,
- the outputs can be regenerated,
- and the plots can be rebuilt from exported CSV data.

For grading or demo purposes, a reduced experiment path is recommended. Larger sweeps are better treated as extended evaluation rather than the default reproduction path.

---

## 14. Future Work

- cycle-accurate architectural simulation,
- hardware-assisted rollback,
- explicit queue and backpressure modeling,
- real power / energy measurement,
- broader GPU-backed benchmark sweeps,
- larger prompt sets and stronger draft-target hierarchies.

---

## 15. References / Starting Points

Useful public references mentioned in the project:

- Karpathy speculative decoding implementation
- vLLM
- Hugging Face Transformers
- llama.cpp
- nanoGPT
- Apache TVM
- PyTorch

These are useful for:

- control flow,
- scheduling,
- verification strategy,
- memory handling,
- runtime design.

---

## 16. Authors

- Yanni Rohan Kommathoti
- Nikhil Peravali
- Rajiv Sai Charan Tirumalasetti
