from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from metrics import structure_score, plan_adherence, hallucination_proxy

app = FastAPI(title="Struct-Align API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_SFT = Path("checkpoints/sft/adapter")
ADAPTER_PPO = Path("checkpoints/ppo/adapter")
_model = None; _tok = None

class PlanRequest(BaseModel):
    prompt: str

class GenerateRequest(BaseModel):
    prompt: str
    outline: List[str] | None = None
    strategy: str = "plan-first"

class ScoreRequest(BaseModel):
    prompt: str
    outline: List[str]
    text: str

class CompareResponse(BaseModel):
    outline: List[str]
    baseline_text: str
    plan_text: str
    metrics_baseline: Dict[str, float]
    metrics_plan: Dict[str, float]

def load_model():
    """Loads base → SFT adapter → PPO adapter (if present)."""
    global _model, _tok
    if _model is not None:
        return _model, _tok
    _tok = AutoTokenizer.from_pretrained(BASE, use_fast=True); _tok.pad_token = _tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    if ADAPTER_SFT.exists():
        m = PeftModel.from_pretrained(m, str(ADAPTER_SFT))
    if ADAPTER_PPO.exists():
        m = PeftModel.from_pretrained(m, str(ADAPTER_PPO))
    _model = m
    return _model, _tok

def extract_outline(text: str):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    outs = [re.sub(r"^\d+\.\s*", "", ln) for ln in lines if re.match(r"^\d+\.", ln)]
    return outs[:6] if outs else []

@app.post("/v1/plan")
def plan(req: PlanRequest):
    m,tok = load_model()
    prompt = req.prompt + "\n\nOnly output an Outline with 3-6 numbered bullets."
    ii = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ii, max_new_tokens=100, do_sample=False)
    outline = extract_outline(tok.decode(out[0], skip_special_tokens=True)) or ["Introduction","Method","Results"]
    return {"outline": outline}

@app.post("/v1/generate")
def generate(req: GenerateRequest):
    m,tok = load_model()
    if req.strategy == "baseline":
        q = tok(req.prompt, return_tensors="pt").to(m.device)
        out = m.generate(**q, max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.9)
        return {"text": tok.decode(out[0], skip_special_tokens=True)}
    outline = req.outline or plan(PlanRequest(prompt=req.prompt))["outline"]
    gen_prompt = f"{req.prompt}\n\nUse this Outline:\n" + "\n".join([f"{i+1}. {o}" for i,o in enumerate(outline)])
    q = tok(gen_prompt, return_tensors="pt").to(m.device)
    out = m.generate(**q, max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.9)
    return {"outline": outline, "text": tok.decode(out[0], skip_special_tokens=True)}

@app.post("/v1/score")
def score(req: ScoreRequest):
    return {
        "structure": structure_score(req.text),
        "plan_adherence": plan_adherence(req.outline, req.text),
        "hallucination_proxy": hallucination_proxy(req.text),
    }

@app.post("/v1/compare", response_model=CompareResponse)
def compare(req: PlanRequest):
    o = plan(req)["outline"]
    base = generate(GenerateRequest(prompt=req.prompt, outline=o, strategy="baseline"))["text"]
    plan_txt = generate(GenerateRequest(prompt=req.prompt, outline=o, strategy="plan-first"))["text"]
    mb = score(ScoreRequest(prompt=req.prompt, outline=o, text=base))
    mp = score(ScoreRequest(prompt=req.prompt, outline=o, text=plan_txt))
    return CompareResponse(
        outline=o, baseline_text=base, plan_text=plan_txt,
        metrics_baseline=mb, metrics_plan=mp
    )