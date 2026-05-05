# Speculative Decoding and Multi-Token Pipelines

A software prototype and architecture study of **speculative decoding** for large language model inference, built for **CECS 530 / Advanced Computer Architecture**.

This project studies speculative decoding as more than a text-generation optimization. It treats the problem as a **pipeline, control, and memory-system design problem**: draft generation, target verification, token commit, rollback, backpressure, and KV-cache management are modeled as interacting stages of a multi-token inference pipeline.

---

## Reproduction Note

This repository provides two Git branches for different evaluation needs:

- **`demo-48-run-sweep`**  
  A reduced configuration with **48 checks** that completes in **less than 2 minutes** on a typical local machine. This branch is recommended for quick grading, demo walkthroughs, and reproducibility verification.

  [Go to `demo-48-run-sweep` branch](https://github.com/RohanK-coder/Speculative-Decoding-and-Multi-Token-Pipelines/tree/demo-48-run-sweep)

- **`main`**  
  The full configuration with **960 checks** that takes about **20 minutes** on a typical local machine. This branch is intended for more complete evaluation and broader experimental analysis.

  [Go to `main` branch](https://github.com/RohanK-coder/Speculative-Decoding-and-Multi-Token-Pipelines/tree/main)

Both branches include generated outputs and plots so that results can be inspected directly and regenerated from the experiment scripts.

For quick verification, use **`demo-48-run-sweep`**. For extended experiments, use **`main`**.

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

- **Algorithmic correctness** — only verified tokens are committed.
- **Pipeline architecture** — draft, verify, commit, and rollback stages interact like a multi-token pipeline.
- **Systems tradeoffs** — speedup depends on acceptance rate, draft/target cost ratio, speculation depth, verification cost, rollback overhead, and KV-cache behavior.

Key architecture questions explored in this project include:

- How can speculative decoding break the one-token-at-a-time bottleneck of autoregressive generation?
- When does speculation outperform baseline decoding?
- How do acceptance rate, speculation depth, verification cost, rollback, backpressure, and KV-cache management affect end-to-end speed?
- How can adaptive and hybrid control policies make speculative decoding more robust across different prompts and model families?

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
- experiment scripts for single runs, comparisons, and grid sweeps,
- generated CSV/JSON result files,
- generated plots for report and presentation use,
- analytical and empirical views of:
  - speedup,
  - acceptance rate,
  - verification bottlenecks,
  - stall rounds,
  - backpressure,
  - wasted draft work,
  - energy proxy,
  - memory and KV-cache overhead.

---

## 3. Repository Layout

The root-level project structure is:

```text
Speculative-Decoding-and-Multi-Token-Pipelines/
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

> Note: A local `.venv/` directory may exist on the developer machine, but it is intentionally ignored by Git and should be recreated locally by each user.

### Important files and directories

- `app.py`  
  Streamlit interface for baseline vs speculative decoding, live metrics, and interactive demos.

- `core/`  
  Core model wrappers, baseline decoding, speculative decoding, analysis helpers, metrics, and utilities.

- `experiments/run_single.py`  
  Runs one experiment configuration for quick testing.

- `experiments/validate_correctness.py`  
  Validates that speculative decoding only commits verified tokens and handles rollback correctly.

- `experiments/run_grid.py`  
  Batch experiment runner for parameter sweeps.

- `experiments/compare_model_families.py`  
  Compares multiple model families under similar decoding settings.

- `experiments/compare_algorithms.py`  
  Compares baseline and speculative approaches.

- `experiments/export_best_configs.py`  
  Exports best-performing configurations from generated results.

- `experiments/plot_results.py`  
  Generates plots from exported CSV results.

- `outputs/results/`  
  CSV and JSON experiment outputs. These files are included so results can be inspected directly and regenerated.

- `outputs/plots/`  
  Generated figures for the report, slides, and grading review.

- `outputs/logs/`  
  Optional run logs.

- `tests/`  
  Unit and correctness tests for important decoding behavior.

---

## 4. Supported Model Families

The project was designed around small draft/target configurations so that the main workflows can run on local machines.

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

Some larger or gated models may require additional memory, Hugging Face authentication, or license approval.

---

## 5. Environment Setup

### Recommended Python version

Use **Python 3.10–3.12**.

Python 3.13 may work for some parts, but ML package support and `torch` / `torchvision` compatibility can be inconsistent.

### Create and activate a virtual environment

Create a local virtual environment so dependencies are isolated from system-level Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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

### Environment notes

- Run all commands from the **project root**, the folder that contains `app.py`, `core/`, `experiments/`, and `requirements.txt`.
- Activate `.venv` before running the app or experiments.
- Use `PYTHONPATH=.` when running scripts from the command line.
- Some Hugging Face models may require authentication or license approval.
- `meta-llama/Llama-3.2-3B-Instruct` may require a Hugging Face login and accepted terms.
- On limited hardware, use the `gpt2` family first.

---

## 6. Reproducibility Guide

This section gives a complete workflow for launching the app, running experiments, exporting results, and regenerating plots.

The repository is designed to be **functionally reproducible**. The experiment outputs and plots are included in both branches, and the scripts can regenerate those artifacts from the repository root.

### Step 1 — Choose a branch

For quick verification:

```bash
git checkout demo-48-run-sweep
```

For the full experiment configuration:

```bash
git checkout main
```

Use `demo-48-run-sweep` for fast grading checks and `main` for the complete sweep.

### Step 2 — Activate the virtual environment

Before running anything:

```bash
source .venv/bin/activate
```

If the virtual environment does not exist yet, create it first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Run the app

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
- observe metrics such as acceptance rate, rollback behavior, wasted draft work, and speedup.

### Recommended first app run

Use a lightweight setup first:

```text
family: gpt2
mode: hybrid
k: 2 or 3
max new tokens: 32
device: cpu
```

This is the safest first run for correctness and basic performance checks.

### Step 4 — Run the correctness check

```bash
PYTHONPATH=. python experiments/validate_correctness.py
```

This verifies that the speculative decoder commits only validated tokens and handles rollback behavior correctly.

### Step 5 — Run one experiment configuration

For a quick test that does not require a large sweep:

```bash
PYTHONPATH=. python experiments/run_single.py
```

Use this when you want one reproducible example for a demo or grading walkthrough.

### Step 6 — Run comparison scripts

Model-family comparison:

```bash
PYTHONPATH=. python experiments/compare_model_families.py
```

Algorithm comparison:

```bash
PYTHONPATH=. python experiments/compare_algorithms.py
```

These are useful for focused comparisons without running the full grid.

### Step 7 — Run the grid experiment

```bash
PYTHONPATH=. python experiments/run_grid.py
```

This generates structured outputs for multiple combinations of:

- prompts,
- prompt categories,
- speculation depth `k`,
- output length,
- policy mode,
- and speed-ratio cases.

The `demo-48-run-sweep` branch runs a reduced grid for quick reproduction. The `main` branch runs the full grid.

### Step 8 — Check exported results

Experiment outputs are stored under:

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

These generated outputs are included in the repository so the results can be reviewed directly and regenerated using the scripts above.

### Step 9 — Generate plots

After results are exported, generate figures with:

```bash
PYTHONPATH=. python experiments/plot_results.py
```

Typical plot outputs include:

```text
outputs/plots/speedup_vs_k.png
outputs/plots/acceptance_vs_k.png
outputs/plots/speedup_vs_acceptance.png
outputs/plots/speedup_vs_cost_ratio.png
outputs/plots/energy_per_token_proxy.png
outputs/plots/slowdown_combined_realistic.png
```

### Step 10 — Recommended end-to-end workflow

For the cleanest reproduction path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=. python experiments/validate_correctness.py
PYTHONPATH=. python experiments/run_single.py
PYTHONPATH=. python experiments/run_grid.py
PYTHONPATH=. python experiments/plot_results.py
```

For a faster grading workflow, use the `demo-48-run-sweep` branch before running the commands above.

---

## 7. Expected Outputs and Plot Set

The repository includes generated output files and plots so that the results can be inspected without rerunning every experiment.

### CSV / JSON outputs

Recommended logged fields include:

- prompt,
- category,
- max new tokens,
- mode,
- `k`,
- acceptance rate,
- speedup,
- rollback count,
- baseline fallback steps,
- wasted draft tokens,
- draft time,
- verify time,
- commit time,
- rollback penalty,
- pipeline utilization,
- stall rounds,
- backpressure events,
- verify bottleneck ratio,
- energy proxy,
- KV-cache overhead proxy,
- output match.

### Recommended plots

1. **Speedup vs speculation depth `k`**  
   Shows latency tradeoffs and diminishing returns.

2. **Acceptance rate vs `k`**  
   Shows how deeper speculation affects accepted work.

3. **Speedup vs acceptance rate**  
   Shows the strongest causal relationship behind speculation efficiency.

4. **Speedup vs draft/target cost ratio**  
   Shows why a cheap draft model matters.

5. **Energy-per-token proxy vs `k`**  
   Shows that the fastest setting is not always the most efficient one.

6. **Slowdown analysis plots**  
   Diagnose verification bottlenecks, backpressure, stall rounds, and wasted draft tokens.

---

## 8. Suggested Commands

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

### Grid sweep

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

## 9. Expected Findings

Across the project, the main expected conclusions are:

- speculative decoding is **not always faster** than baseline,
- **acceptance rate** is the most important performance indicator,
- speculation depth `k` has an **optimal middle range**,
- very large `k` can cause diminishing returns because rollback and verification overhead increase,
- a **cheap draft model** is important for useful speedup,
- adaptive and hybrid control policies improve robustness,
- speedup alone is not enough; memory overhead and energy proxy should also be considered.

In one sentence:

> Speculative decoding outperforms baseline only when accepted speculative work is large enough to amortize verification, rollback, and control overhead.

---

## 10. Limitations

- Some larger model families require more memory than a typical laptop provides.
- Gated models may require Hugging Face authentication and accepted license terms.
- CPU-only runs are supported but may be slower for larger sweeps.
- A subset of presentation figures may be trend illustrations rather than large-scale measured sweeps.
- Energy results are proxy-based unless collected from actual device instrumentation.
- The full grid experiment can be time-consuming on CPU-only systems.
- Exact package versions are not pinned in this submission, but the project is designed to run inside a local virtual environment using `requirements.txt`.

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'core'`

Run scripts from the project root with `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

### `python3.11: command not found`

Use the Python version installed on your system:

```bash
python3 --version
python3 -m venv .venv
```

### `.venv/bin/activate: no such file or directory`

The virtual environment has not been created yet. Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### `python: command not found`

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Then check:

```bash
python --version
```

### Hugging Face model download or authentication issues

Start with `distilgpt2` and `gpt2`. If you later use gated models, make sure your Hugging Face login and accepted terms are configured.

### Plot script cannot find result files

Run the grid experiment first:

```bash
PYTHONPATH=. python experiments/run_grid.py
```

Then regenerate plots:

```bash
PYTHONPATH=. python experiments/plot_results.py
```

Also confirm that result files exist under:

```text
outputs/results/
```

---

## 12. Reproducibility Statement

This repository is designed to be **functionally reproducible**:

- the decoding logic can be rerun,
- correctness checks can be repeated,
- the experiments can be reswept,
- the generated outputs are included,
- the plots are included,
- and the figures can be rebuilt from exported CSV data.

For grading or demo purposes, use the reduced branch:

```bash
git checkout demo-48-run-sweep
```

For complete evaluation, use:

```bash
git checkout main
```

The reduced branch is intended for quick verification. The full branch is intended for broader analysis.

---

## 13. Future Work

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

## 14. References / Starting Points

Useful public references and systems related to this project include:

- speculative decoding papers and implementations,
- Karpathy speculative decoding examples,
- Hugging Face Transformers,
- vLLM,
- llama.cpp,
- nanoGPT,
- Apache TVM,
- PyTorch.

These references are useful for understanding:

- decoding control flow,
- verification strategy,
- scheduling,
- rollback behavior,
- memory handling,
- runtime design,
- and inference-system tradeoffs.

---

## 15. Authors

- Yanni Rohan Kommathoti
- Nikhil Peravali
- Rajiv Sai Charan Tirumalasetti
