# Structued Alignment LM — Plan-First LLM (LoRA SFT + mini-PPO)

Fine-tunes a small LLM to **plan first (Outline) → then write**, and reinforces that behavior with a **structure-aware reward**. Includes a **FastAPI backend** and **Next.js UI** that compare *baseline* vs *plan-first* outputs with simple metrics.

- Built an **end-to-end LLM alignment** system that makes the model **plan first (Outline) → then write** using **LoRA SFT + mini-PPO (RLAIF)** on **TinyLlama-1.1B**; adapters target **Q/K/V/O** projections and run comfortably on a **single RTX 4070 (≈8 GB VRAM)**.
- **Curated structure-aligned data** from PubMed RCT abstracts into *(prompt, outline→sections)* JSONL pairs to teach discourse scaffolding during SFT.
- Implemented a **dense reward** `0.6·structure_score + 0.4·plan_adherence`; during PPO, reward rose **0.643 → 0.683**, and **mean structure improved 0.061 → 0.403 (≈6.6×)** on held-out prompts.
- Shipped a **FastAPI backend** (`/v1/plan`, `/v1/generate`, `/v1/score`, `/v1/compare`) that serves **base → SFT adapter → PPO adapter** at inference (no base merges).
- Built a **Next.js (App Router) UI** to A/B **baseline vs plan-first** with real-time metrics, demo prompts, CORS, and env-configurable API base.
- Engineered a **reproducible CUDA environment** with **uv (Python 3.11)** + **PyTorch cu121**, resolved Windows/PowerShell quirks (e.g., `TRANSFORMERS_NO_TORCHVISION`), and kept training/inference efficient.

**Tech:** PyTorch, HF Transformers, TRL, PEFT, Datasets, FastAPI, Uvicorn, Next.js/React/TypeScript, `uv`, CUDA.

---

## How it works

1. **Data → “plan then write” pairs**  
   Training examples tell the model to (a) produce a 3–6 bullet **Outline**, then (b) write one **numbered section per bullet**. Responses contain `Outline:` plus the sectioned text. This teaches the *habit* of planning before writing.

2. **LoRA SFT (supervised fine-tuning)**  
   We attach **LoRA adapters** to attention projections (Q/K/V/O) and train only a tiny low-rank **ΔW** that’s *added at runtime* to the frozen base. This is fast, memory-light, and preserves base knowledge.

3. **mini-PPO (RLAIF) with a structure reward**  
   For each prompt: generate an **Outline** (greedy) → generate the **Sectioned answer** (sampled) → compute  
   `reward = 0.6 * structure_score + 0.4 * plan_adherence`.  
   PPO nudges policy toward higher reward. PPO adapter stacks on top of SFT.

4. **Serving & UI**  
   FastAPI loads base → SFT adapter → PPO adapter. Endpoints: `/v1/plan`, `/v1/generate`, `/v1/score`, `/v1/compare`.  
   Next.js UI calls `/v1/compare`, shows both texts & metrics, and includes demo prompts.

---

## Repo layout

    app/
      build_dataset.py        # Create (prompt, response) pairs for outline→sections SFT
      train_sft.py            # LoRA SFT; saves adapter to checkpoints/sft/adapter
      train_ppo.py            # PPO with structure reward; saves adapter to checkpoints/ppo/adapter
      batch_eval.py           # Quick A/B metrics + a bar plot (baseline vs plan-first)
      metrics.py              # structure_score, plan_adherence, hallucination_proxy
      main.py                 # FastAPI: /v1/plan, /v1/generate, /v1/score, /v1/compare
      checkpoints/
        sft/adapter           # created by train_sft.py
        ppo/adapter           # created by train_ppo.py
    frontend/
      app/page.tsx            # Next.js UI (demo chips + compare panel)
    pyproject.toml            # uv project + dependencies (Python 3.11)
    README.md

---

## Prerequisites

- **Windows + PowerShell**, **VS Code**
- **NVIDIA GPU** + driver (check with `nvidia-smi`)
- **Python** managed by **uv** → https://docs.astral.sh/uv/
- **Node 18+** (for the Next.js app)
- Internet access for Hugging Face downloads

> Windows note: Hugging Face may warn about symlink cache; it still works (degraded cache).

---

## 1) Environment setup

From repo root (where `pyproject.toml` lives):

    uv venv
    uv sync

**Verify CUDA in the venv**

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
    # Expect: is_available True and a CUDA version (e.g., 12.1)

If CUDA is `False`, see **Troubleshooting**.

---

## 2) Data → SFT → PPO

### A) Build dataset

Creates `data/struct_train.jsonl` & `data/struct_val.jsonl` with “Outline → Sections” samples.

    uv run python app/build_dataset.py

### B) Supervised fine-tuning (LoRA)

Learns the plan-first habit.

    uv run python app/train_sft.py
    # Output: checkpoints/sft/adapter

### C) mini-PPO (structure reward)

Reinforces structure and plan adherence:

- `structure_score(text)` — headings/transition cues + cohesion  
- `plan_adherence(outline, text)` — coverage + order consistency

    uv run python app/train_ppo.py
    # Output: checkpoints/ppo/adapter
    # Console prints: step N reward=0.xxx (should trend up modestly)

---

## 3) Run the API & UI

### A) Start FastAPI

    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
    # Swagger UI: http://localhost:8000/docs

### B) Start Next.js

    cd frontend
    # Optional: set API base (defaults to http://localhost:8000)
    # PowerShell:
    $env:NEXT_PUBLIC_API_BASE = "http://localhost:8000"
    npm i
    npm run dev
    # Open http://localhost:3000

---

## 4) Smoke-tests

**Plan only (PowerShell):**

    Invoke-RestMethod http://localhost:8000/v1/plan -Method POST `
      -ContentType "application/json" `
      -Body (@{prompt="Draft a research-methods section for note-taking apps"} | ConvertTo-Json)

**Full compare (PowerShell):**

    Invoke-RestMethod http://localhost:8000/v1/compare -Method POST `
      -ContentType "application/json" `
      -Body (@{prompt="Write a policy brief on urban heat mitigation"} | ConvertTo-Json) | ConvertTo-Json

**In the UI:** paste your prompt or click a **Demo** chip → **Submit & Compare**.

---

## 5) Batch evaluation (optional)

Compare mean structure across several prompts and save a plot.

    uv run python app/batch_eval.py
    # Outputs:
    # - data/report.json (per-prompt metrics)
    # - data/struct_bar.png (mean structure; plan-first should be ↑)

---

## Design details

- **Why LoRA:** train tiny low-rank adapters (A/B) that create a **ΔW** added to the frozen base weight → efficient and preserves knowledge.
- **QLoRA vs LoRA:** QLoRA = base in 4-bit + LoRA adapters. Here we use standard **LoRA** on a 1.1B base (fits on consumer GPUs) to keep setup simple.
- **Adapters stacking:** base → **SFT adapter** → **PPO adapter** applied at load time (no heavy merges).
- **Reward signals:** purely local, no external judge—fast and reproducible.

---

## Endpoints

- **POST `/v1/plan`**  
  **Body:** `{ "prompt": "..." }`  
  **Resp:** `{ "outline": ["..."] }`

- **POST `/v1/generate`**  
  **Body:** `{ "prompt": "...", "outline"?: [...], "strategy": "baseline" | "plan-first" }`  
  **Resp:** `{ "text": "..." }` (and `outline` if plan-first)

- **POST `/v1/score`**  
  **Body:** `{ "prompt": "...", "outline": [...], "text": "..." }`  
  **Resp:** `{ "structure": x, "plan_adherence": y, "hallucination_proxy": z }`

- **POST `/v1/compare`**  
  **Body:** `{ "prompt": "..." }`  
  **Resp:** `{ outline, baseline_text, plan_text, metrics_baseline, metrics_plan }`

---

## Troubleshooting

**CUDA shows False / CPU wheel installed**

    # Reinstall CUDA wheels (example for cu121)
    uv pip uninstall torch torchvision torchaudio
    uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision torchaudio
    uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

**Transformers importing torchvision (text-only)**

    $env:TRANSFORMERS_NO_TORCHVISION = "1"

**Hallucination score seems “high”**  
The proxy penalizes **uncited numbers**. Lower it by:
- Prompting “avoid specific statistics / avoid numbers,” or
- Adding simple `[ref]` tags when numbers appear, or
- Reducing numeral weight in `app/metrics.py`.

**Slow / OOM**  
Lower `max_new_tokens` (e.g., 650 → 320) in `train_ppo.py`, `batch_eval.py`, and `app/main.py`.

---

## Notes
- Remove `checkpoints/ppo/adapter` to ablate PPO and see pure SFT behavior.
- To try a larger base (e.g., 3B), update the `BASE` constant and ensure VRAM is sufficient.
