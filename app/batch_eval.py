import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from metrics import structure_score, plan_adherence, topic_drift, hallucination_proxy

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
AD_SFT = "checkpoints/sft/adapter"
AD_PPO = "checkpoints/ppo/adapter"

PROMPTS = [
    "Explain tradeoffs of transformer depth vs width for long-form writing.",
    "Design a week-long lesson plan to teach recursion to high schoolers.",
    "Write a policy brief on urban heat mitigation (with outline & sections).",
    "Draft a research-methods section for a usability study on note-taking apps.",
    "Create a troubleshooting guide for a failing CI pipeline.",
    "Outline and write a specific aims page for a wildfire forecasting grant."
]

def extract_outline(text: str):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    outs = [re.sub(r"^\d+\.\s*", "", ln) for ln in lines if re.match(r"^\d+\.", ln)]
    return outs[:6] if outs else []

def load_model():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
    m = PeftModel.from_pretrained(m, AD_SFT)
    m = PeftModel.from_pretrained(m, AD_PPO)
    m.eval()
    return m, tok

def gen_plan_first(m, tok, prompt):
    ip = tok(prompt + "\n\nOnly output an Outline with 3-6 numbered bullets.",
             return_tensors="pt").to(m.device)
    ot = m.generate(**ip, max_new_tokens=90, do_sample=False)
    outline = extract_outline(tok.decode(ot[0], skip_special_tokens=True)) or ["Introduction","Method","Results"]
    q = tok(prompt + "\n\nUse this Outline:\n" + "\n".join(f"{i+1}. {o}" for i,o in enumerate(outline)),
            return_tensors="pt").to(m.device)
    out = m.generate(**q, max_new_tokens=650, do_sample=True, temperature=0.55, top_p=0.9)
    return outline, tok.decode(out[0], skip_special_tokens=True)

def gen_baseline(m, tok, prompt):
    q = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**q, max_new_tokens=650, do_sample=True, temperature=0.55, top_p=0.9)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    Path("data").mkdir(exist_ok=True)
    m, tok = load_model()
    rows = []
    for p in PROMPTS:
        outline, plan_txt = gen_plan_first(m, tok, p)
        base_txt = gen_baseline(m, tok, p)
        mb = dict(
            structure=structure_score(base_txt),
            plan=plan_adherence(outline, base_txt),
            drift=topic_drift(outline, base_txt),
            halluc=hallucination_proxy(base_txt),
        )
        mp = dict(
            structure=structure_score(plan_txt),
            plan=plan_adherence(outline, plan_txt),
            drift=topic_drift(outline, plan_txt),
            halluc=hallucination_proxy(plan_txt),
        )
        rows.append(dict(prompt=p, outline=outline, baseline=mb, planfirst=mp))
    Path("data/report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    s_base = float(np.mean([r["baseline"]["structure"] for r in rows]))
    s_plan = float(np.mean([r["planfirst"]["structure"] for r in rows]))
    plt.figure()
    plt.title("Mean Structure Score (↑ better)")
    plt.bar(["baseline","plan-first"], [s_base, s_plan])
    plt.ylabel("score (0–1)")
    plt.savefig("data/struct_bar.png", dpi=150, bbox_inches="tight")

    print("Wrote data/report.json and data/struct_bar.png")
    print(f"Mean structure: baseline={s_base:.3f} plan-first={s_plan:.3f}")

if __name__ == "__main__":
    main()