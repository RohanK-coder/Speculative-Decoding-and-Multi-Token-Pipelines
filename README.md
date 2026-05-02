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

## 6. Running the App, Generating Outputs, and Reproducing Plots

This section gives one clear workflow for launching the app, running experiments, exporting results, and generating plots.

### Step 1 — Activate the virtual environment

Before running anything, activate the project virtual environment:

```bash
source .venv/bin/activate
```

If you have not created it yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2 — Launch the interactive app

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The app is the easiest way to verify that the project is working.

Inside the app, you can:

- choose a model family,
- compare baseline decoding and speculative decoding,
- vary speculation depth `k`,
- select control policies such as `fixed`, `adaptive`, or `hybrid`,
- observe live metrics such as acceptance rate, rollback behavior, and speedup.

Recommended first run:

- family: `gpt2`
- mode: `hybrid`
- `k = 2` or `k = 3`
- output length: `32`

This lightweight setup is the best starting point for confirming that the decoding flow works correctly before running larger experiments.

---

### Step 3 — Validate correctness with a small command-line run

After confirming the app launches, run a small command-line comparison:

```bash
python specupipe/experiments/compare_model_families.py \
  --family gpt2 \
  --question "What is speculative decoding?" \
  --max_new_tokens 32 \
  --k 2 \
  --device cpu \
  --mode fixed
```

Use this step to confirm that:

- the baseline path runs correctly,
- the speculative path only commits verified tokens,
- rejected suffixes are rolled back properly,
- the final committed output remains consistent with target-model greedy decoding.

You can also try additional runs such as:

```bash
python specupipe/experiments/compare_model_families.py \
  --family tinyllama \
  --question "Describe rollback in speculative decoding." \
  --max_new_tokens 48 \
  --k 2 \
  --device cpu \
  --mode hybrid
```

and

```bash
python specupipe/experiments/compare_model_families.py \
  --family smollm2 \
  --question "Explain why speculative decoding can fail." \
  --max_new_tokens 32 \
  --k 3 \
  --device cpu \
  --mode adaptive
```

---

### Step 4 — Run the full experiment sweep

To generate reproducible performance outputs across multiple settings, run:

```bash
python specupipe/experiments/run_grid.py
```

This is the main experiment driver for sweeping parameters such as:

- speculation depth `k`,
- output length,
- model family,
- control mode,
- and prompt category.

A typical sweep should vary values such as:

- `k = 2, 3, 4, 6, 8`
- output lengths like `32` and `64`
- policies like `fixed`, `adaptive`, `hybrid`, and `category-aware`

---

### Step 5 — Check exported outputs

Experiment results should be saved to output directories such as:

```text
outputs/csv/
outputs/logs/
outputs/plots/
```

Recommended exported fields include:

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

The most important file for plotting is typically a CSV such as:

```text
outputs/csv/results.csv
```

If your script writes a differently named file, use that file path in the plotting step.

---

### Step 6 — Generate plots from the exported CSV

After the sweep completes and the CSV has been written, generate plots with a plotting script such as:

```bash
python specupipe/experiments/plot_results.py --input outputs/csv/results.csv --outdir outputs/plots
```

This should generate figures such as:

```text
outputs/plots/speedup_vs_k.png
outputs/plots/acceptance_vs_k.png
outputs/plots/speedup_vs_acceptance.png
outputs/plots/speedup_vs_cost_ratio.png
outputs/plots/energy_per_token_proxy.png
outputs/plots/slowdown_combined_realistic.png
```

If `plot_results.py` is not yet included in the repository, add a small plotting utility that reads the exported CSV and produces the main figures.

---

### Step 7 — Recommended plot set

The most useful plots for this project are:

1. **Speedup vs speculation depth `k`**  
   Shows the main latency tradeoff and diminishing returns.

2. **Acceptance rate vs `k`**  
   Explains why speedup rises or falls as speculation depth changes.

3. **Speedup vs acceptance rate**  
   Shows the strongest relationship behind speculation efficiency.

4. **Speedup vs draft/target cost ratio**  
   Shows why a cheaper draft model is important.

5. **Energy-per-token proxy vs `k`**  
   Shows that the fastest setting is not always the most efficient one.

6. **Slowdown analysis plots**  
   Useful for understanding why performance degrades:
   - verify bottleneck,
   - backpressure,
   - stall rounds,
   - wasted draft tokens.

---

### Step 8 — Recommended end-to-end workflow

For the clearest reproduction path, use this order:

1. create and activate the virtual environment,
2. launch the app with `streamlit run specupipe/app.py`,
3. test a small `gpt2` baseline vs speculative run,
4. run `compare_model_families.py` for sanity checking,
5. run `run_grid.py` for the full sweep,
6. confirm CSV/log output files were created,
7. run the plotting script to generate figures in `outputs/plots/`.

---

### Reproducibility note

This repository is designed to be functionally reproducible:

- the decoding logic can be rerun,
- the experiments can be reswept,
- the CSV outputs can be regenerated,
- and the plots can be rebuilt from exported results.

Some presentation figures may be illustrative trend plots rather than full measured sweeps. For strict measurement-based reporting, rerun the experiments on a stronger CPU or GPU environment and regenerate all figures directly from exported CSV data.

---

## 7. Expected Findings

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

## 8. Limitations

- Some larger model families require more memory than a typical laptop provides.
- Gated models may require authentication.
- A subset of presentation figures may be trend illustrations rather than large-scale measured sweeps.
- Energy results are proxy-based unless collected from actual device instrumentation.

---

## 9. Future Work

- cycle-accurate architectural simulation,
- hardware-assisted rollback,
- explicit queue and backpressure modeling,
- real power / energy measurement,
- broader GPU-backed benchmark sweeps,
- larger prompt sets and stronger draft-target hierarchies.

---

## 10. References / Starting Points

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

## 11. Quick Start

If you only want to see the project working quickly:

```bash
python3 -m venv .venv
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

## 12. Reproducibility Statement

This repository is designed to be **functionally reproducible**:

- the decoding logic can be rerun,
- the experiments can be reswept,
- the plots can be regenerated,
- and presentation figures can be replaced with measured outputs on stronger hardware.

Where presentation plots are illustrative rather than fully measured, that fact should be stated explicitly in the report or slides.

---

## 13. Authors

- Yanni Rohan Kommathoti
- Nikhil Peravali
- Rajiv Sai Charan Tirumalasetti
