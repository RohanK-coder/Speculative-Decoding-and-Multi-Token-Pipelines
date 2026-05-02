# Speculative Decoding and Multi-Token Pipelines

A software prototype and architecture study of **speculative decoding** for large language model inference, built for **CECS 530 / Advanced Computer Architecture**.

This project treats speculative decoding as more than a text-generation trick. It studies the problem as a **pipeline, control, and memory-system design problem**:

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
- a **web dashboard / app** for interactive runs,
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

> The exact file names below match the working project structure used during development. If your local clone differs slightly, adjust the paths accordingly.

```text
specupipe/
├── app.py
├── core/
│   ├── models.py
│   ├── prompting.py
│   ├── streaming_decode.py
│   └── ...
├── experiments/
│   ├── run_grid.py
│   ├── compare_model_families.py
│   └── ...
├── outputs/
│   ├── plots/
│   ├── csv/
│   └── logs/
├── requirements.txt
└── README.md
```

### Important files

- `specupipe/app.py`  
  Streamlit interface for baseline vs speculative decoding, live metrics, and interactive demos.

- `specupipe/core/models.py`  
  Model and tokenizer loading, forward calls, and helper wrappers.

- `specupipe/core/prompting.py`  
  Prompt formatting per model family.

- `specupipe/core/streaming_decode.py`  
  Streaming baseline/speculative decode logic and runtime metric emission.

- `specupipe/experiments/run_grid.py`  
  Batch experiment runner for parameter sweeps.

- `specupipe/experiments/compare_model_families.py`  
  Compare model families under the same decoding settings.

---

## 4. Supported Model Families

The project was configured around the following model families:

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

---

## 5. Environment Setup

### Recommended Python version

Use **Python 3.10–3.12**.

Python 3.13 may work for some parts, but ML package support and `torch` / `torchvision` compatibility can be inconsistent.

### Create and activate a virtual environment

Using a virtual environment is recommended so dependencies remain isolated to this project and are **not installed globally**.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies inside the virtual environment

If your repository includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you need to install core dependencies manually:

```bash
pip install torch torchvision transformers streamlit matplotlib pandas numpy sentencepiece accelerate
```

### Notes

- Install all packages **inside the activated virtual environment**, not system-wide.
- Some Hugging Face models may require authentication or license approval.
- `meta-llama/Llama-3.2-3B-Instruct` may require a Hugging Face login and accepted terms.
- On limited hardware, use smaller families first (`gpt2`, `tinyllama`) before trying larger targets.

---

## 6. Running the Project

### Launch the interactive app

```bash
streamlit run specupipe/app.py
```

This opens the dashboard for:

- baseline decoding,
- speculative decoding,
- policy selection,
- model-family selection,
- and live metrics display.

### Run a command-line comparison

```bash
python specupipe/experiments/compare_model_families.py \
  --family gpt2 \
  --question "Explain speculative decoding in simple words." \
  --max_new_tokens 32 \
  --k 3 \
  --device cpu \
  --mode hybrid
```

### Run a parameter sweep / grid experiment

```bash
python specupipe/experiments/run_grid.py
```

This is the main entry point for reproducing performance sweeps across:

- speculation depth `k`,
- output length,
- policy type,
- and model-family choice.

---

## 7. Reproducibility Guide

This section explains the recommended order to follow if you want to reproduce the experiments and presentation outputs.

### Step 1 — Verify the environment

Make sure your virtual environment is activated first:

```bash
source .venv/bin/activate
python -c "import torch, transformers, streamlit, matplotlib; print('env ok')"
```

### Step 2 — Start with a lightweight family

Use the smallest pair first to verify correctness and flow:

- draft: `distilgpt2`
- target: `gpt2`

Why: it is faster to download, simpler to debug, and safer on limited hardware.

### Step 3 — Validate correctness

Run a short generation with baseline and speculative modes and confirm:

- the speculative decoder only commits verified tokens,
- mismatched suffixes are rolled back,
- the final committed prefix remains consistent with baseline target greedy decoding.

### Step 4 — Run sweeps over `k`

Use a fixed prompt set and vary:

- `k = 2, 3, 4, 6, 8`
- output lengths such as `32` and `64`
- policies such as `fixed`, `adaptive`, `hybrid`, `category-aware`

### Step 5 — Export CSV or logs

Store outputs in a reproducible directory such as:

```text
outputs/csv/
outputs/logs/
outputs/plots/
```

Recommended fields:

- family
- draft model
- target model
- prompt category
- mode
- `k`
- acceptance rate
- speedup
- rollback count
- verify bottleneck ratio
- stall rounds
- backpressure events
- energy proxy
- memory / KV overhead proxy

### Step 6 — Generate figures from exported data

Use the plotting scripts or notebooks described below.

### Important note for reproducibility

The project still provides a clear path to regenerate the plots on a stronger machine or cloud environment. In other words:

- the **code path is reproducible**,
- but some presentation figures are **representative trend plots** rather than full-scale measured sweeps.

If you want strict measured results only, rerun the sweep scripts on a GPU-enabled environment and replace the illustrative figures with exported CSV-based plots.

---

## 8. How to Generate the Plots

There are two recommended ways.

### Option A — Generate plots from experiment sweeps

1. Run the sweep:

```bash
python specupipe/experiments/run_grid.py
```

2. Make sure it exports a CSV to something like:

```text
outputs/csv/results.csv
```

3. Run the plotting script or notebook.

If you keep a dedicated plot script, a clean command is:

```bash
python specupipe/experiments/plot_results.py --input outputs/csv/results.csv --outdir outputs/plots
```

> If `plot_results.py` is not yet in the repository, add one small plotting utility to read the CSV and generate:
> - speedup vs `k`
> - acceptance vs `k`
> - speedup vs acceptance rate
> - speedup vs draft/target cost ratio
> - energy proxy vs `k`

### Option B — Generate presentation-ready illustrative plots

If the repository includes a Colab notebook or local notebook, you can run the plotting cells directly to regenerate the presentation-style figures.

Typical outputs:

```text
outputs/plots/speedup_vs_k.png
outputs/plots/acceptance_vs_k.png
outputs/plots/speedup_vs_acceptance.png
outputs/plots/speedup_vs_cost_ratio.png
outputs/plots/energy_per_token_proxy.png
outputs/plots/slowdown_combined_realistic.png
```

---

## 9. Recommended Plot Set

For this project, the most useful plots are:

1. **Speedup vs speculation depth `k`**  
   Shows the main latency tradeoff and diminishing returns.

2. **Acceptance rate vs `k`**  
   Explains why speedup rises or falls.

3. **Speedup vs acceptance rate**  
   Shows the strongest causal relationship behind speculation efficiency.

4. **Speedup vs draft/target cost ratio**  
   Explains why a cheap draft model matters.

5. **Energy-per-token proxy vs `k`**  
   Shows that the fastest setting is not always the most efficient one.

6. **Slowdown analysis plots**  
   Useful for “Why does the run become slow?” slides:
   - verify bottleneck,
   - backpressure,
   - stall rounds,
   - wasted draft tokens.

---

## 10. Suggested Reproduction Commands

### Baseline sanity check

```bash
python specupipe/experiments/compare_model_families.py \
  --family gpt2 \
  --question "What is speculative decoding?" \
  --max_new_tokens 32 \
  --k 2 \
  --device cpu \
  --mode fixed
```

### Adaptive-depth run

```bash
python specupipe/experiments/compare_model_families.py \
  --family smollm2 \
  --question "Explain why speculative decoding can fail." \
  --max_new_tokens 32 \
  --k 3 \
  --device cpu \
  --mode adaptive
```

### Hybrid-gating run

```bash
python specupipe/experiments/compare_model_families.py \
  --family tinyllama \
  --question "Describe rollback in speculative decoding." \
  --max_new_tokens 48 \
  --k 2 \
  --device cpu \
  --mode hybrid
```

### Full sweep

```bash
python specupipe/experiments/run_grid.py
```

---

## 11. Expected Findings

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

## 12. Limitations

- Some larger model families require more memory than a typical laptop provides.
- Gated models may require authentication.
- A subset of presentation figures may be trend illustrations rather than large-scale measured sweeps.
- Energy results are proxy-based unless collected from actual device instrumentation.

---

## 13. Future Work

- cycle-accurate architectural simulation,
- hardware-assisted rollback,
- explicit queue and backpressure modeling,
- real power / energy measurement,
- broader GPU-backed benchmark sweeps,
- larger prompt sets and stronger draft-target hierarchies.

---

## 14. References / Starting Points

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

## 15. Quick Start

If you only want to see the project working quickly:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run specupipe/app.py
```

Then:

1. choose a small family (`gpt2`),
2. run baseline and speculative decoding,
3. vary `k`,
4. compare acceptance rate and speedup,
5. export experiments and regenerate plots.

---

## 16. Reproducibility Statement

This repository is designed to be **functionally reproducible**:

- the decoding logic can be rerun,
- the experiments can be reswept,
- the plots can be regenerated,
- and presentation figures can be replaced with measured outputs on stronger hardware.

Where presentation plots are illustrative rather than fully measured, that fact should be stated explicitly in the report or slides.

---

## 17. Authors

- Yanni Rohan Kommathoti
- Nikhil Peravali
- Rajiv Sai Charan Tirumalasetti
