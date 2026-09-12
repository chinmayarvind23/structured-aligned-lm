# Structured Alignment LM — Plan-First LLM with LoRA SFT + PPO

A small-LLM alignment project that trains **TinyLlama-1.1B** to plan before writing, then reinforces that behavior with a structure-aware PPO objective.

The system uses **LoRA supervised fine-tuning (SFT)** to teach an `Outline -> Sections` response format, then applies **mini-PPO / RLAIF** with a dense reward based on structure quality and plan adherence. A FastAPI backend and Next.js UI make it possible to compare baseline and plan-first generations side by side.

## Highlights

- Fine-tuned **TinyLlama-1.1B** with LoRA adapters on Q/K/V/O attention projections, keeping the base model frozen and training on a single **RTX 4070 with about 8 GB VRAM**.
- Built structure-aligned training data from PubMed randomized-controlled-trial abstracts, mapping prompts to explicit outlines and corresponding numbered sections.
- Increased mean held-out **structure score from 0.061 to 0.403, about a 6.6x improvement**, while PPO reward increased from **0.643 to 0.683**.
- Served the full **base -> SFT adapter -> PPO adapter** stack through FastAPI without merging adapters into the base model.
- Built a Next.js comparison UI for baseline-vs-plan-first generation with live structure and plan-adherence metrics.
- Kept the training and inference path reproducible with PyTorch, Hugging Face Transformers, TRL, PEFT, `uv`, and CUDA.

## Why this matters

Small language models can often produce acceptable prose while drifting away from a requested structure or omitting planned sections. This project tests whether explicit planning behavior can be taught and reinforced without moving to a larger model.

The strongest observed result is the held-out structure improvement:

| Metric | Baseline / earlier stage | PPO-aligned result |
| --- | ---: | ---: |
| Mean structure score | **0.061** | **0.403** |
| Relative change |  | **~6.6x** |
| PPO reward | **0.643** | **0.683** |

The result is intentionally narrow: it demonstrates improved structural behavior under this project's metrics and held-out prompts. It is not a claim that the aligned model is universally better at factuality, reasoning, or general language quality.

## What the model learns

Training examples require the model to:

1. produce a short **3–6 item outline**;
2. write one numbered section corresponding to each outline item;
3. preserve the planned ordering in the final response.

The project then measures two main behaviors:

- **structure score** — whether the answer exhibits the expected sectioning, headings, transitions, and cohesion;
- **plan adherence** — whether the generated sections cover the outline and preserve its order.

PPO uses the local dense reward:

```text
reward = 0.6 * structure_score + 0.4 * plan_adherence
```

This makes the optimization target inspectable and reproducible rather than relying on an external black-box judge.

## Training pipeline

```text
PubMed RCT abstracts
        |
        v
Outline -> Sections training pairs
        |
        v
LoRA supervised fine-tuning
        |
        v
SFT adapter
        |
        v
Plan-first generation + local reward
        |
        v
mini-PPO / RLAIF
        |
        v
PPO adapter
        |
        v
FastAPI comparison service
```

### 1. Data construction

`app/build_dataset.py` creates structure-aligned JSONL examples. Each response contains an explicit `Outline:` followed by numbered sections corresponding to the plan.

### 2. LoRA SFT

`app/train_sft.py` attaches low-rank adapters to the attention Q/K/V/O projections while leaving the TinyLlama base weights frozen.

The design keeps the training footprint small enough for a consumer GPU and preserves the original base model separately from the learned behavior adapters.

### 3. PPO alignment

`app/train_ppo.py` generates an outline, samples the sectioned response, computes the structure/plan reward, and updates a PPO adapter on top of the SFT stage.

The resulting inference stack is:

```text
TinyLlama base -> SFT adapter -> PPO adapter
```

Adapters are loaded at inference time rather than permanently merged into the base model.

## Serving and evaluation

The FastAPI service exposes:

- `POST /v1/plan` — generate an outline;
- `POST /v1/generate` — generate baseline or plan-first text;
- `POST /v1/score` — compute structure, plan-adherence, and hallucination-proxy metrics;
- `POST /v1/compare` — run baseline and plan-first generation side by side.

The Next.js UI calls `/v1/compare` and displays both generations together with their metrics, making the behavior change directly inspectable rather than only reporting an aggregate training number.

Batch evaluation is available through:

```powershell
uv run python app/batch_eval.py
```

which writes:

- `data/report.json`
- `data/struct_bar.png`

## Technology

**PyTorch · Hugging Face Transformers · TRL · PEFT · FastAPI · Next.js · CUDA**

PyTorch and Transformers provide model execution, PEFT supplies LoRA adapters, and TRL implements the PPO stage. FastAPI serves planning/generation/scoring endpoints, while Next.js provides the baseline-vs-aligned comparison interface. `uv` and pinned CUDA-compatible dependencies keep the local training environment reproducible.

## Repository structure

```text
app/
├── build_dataset.py   # outline -> sections SFT dataset
├── train_sft.py       # LoRA supervised fine-tuning
├── train_ppo.py       # PPO with structure-aware reward
├── batch_eval.py      # baseline vs plan-first evaluation
├── metrics.py         # structure, adherence, hallucination proxy
└── main.py            # FastAPI service

checkpoints/
├── sft/adapter
└── ppo/adapter

frontend/
└── app/page.tsx       # Next.js comparison UI
```

## Run locally

### Environment

Requirements:

- Python 3.11
- NVIDIA GPU for the trained-model path
- Node 18+
- `uv`

```powershell
uv venv
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### Build the dataset

```powershell
uv run python app/build_dataset.py
```

### Train SFT

```powershell
uv run python app/train_sft.py
```

### Train PPO

```powershell
uv run python app/train_ppo.py
```

### Start the API

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` and compare baseline and plan-first generations.

## Design choices

- **LoRA instead of full fine-tuning:** limits trainable parameters and keeps the 1.1B model practical on a consumer GPU.
- **Standard LoRA instead of QLoRA:** TinyLlama fits within the available GPU budget, so 4-bit base quantization was not required for this implementation.
- **Adapter stacking:** keeps base, SFT, and PPO stages separable for inspection and ablation.
- **Local reward instead of an external judge:** makes the alignment signal inexpensive and reproducible, while also limiting the claim to the behaviors that reward actually measures.

## Scope and limitations

This project measures **structure and plan adherence**, not general intelligence. The reward is hand-designed and local, so improvements should be interpreted as evidence that the model follows the project's desired response structure more consistently, not as evidence of broad capability improvement.

The TinyLlama-1.1B base was chosen to make the full SFT/PPO workflow feasible on a single consumer GPU. Larger models may behave differently and would require separate evaluation.
