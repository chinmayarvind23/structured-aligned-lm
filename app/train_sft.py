import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA = {"train": "data/struct_train.jsonl", "val": "data/struct_val.jsonl"}

def formatting_batched(examples):
    texts = []
    prompts = examples["prompt"]
    responses = examples["response"]
    for p, r in zip(prompts, responses):
        texts.append(f"<s>[INST] {p} [/INST]\n{r}</s>")
    return texts

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    ds = load_dataset("json", data_files=DATA)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )

    cfg = SFTConfig(
        output_dir="checkpoints/sft",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_seq_length=1024,
        num_train_epochs=1,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available(),
        report_to=["none"],
    )

    trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    peft_config=lora,
    formatting_func=formatting_batched,
    args=cfg,
    )
    
    trainer.train()
    trainer.model.save_pretrained("checkpoints/sft/adapter")
    tok.save_pretrained("checkpoints/sft/tokenizer")
    print("SFT done -> checkpoints/sft/adapter")

if __name__ == "__main__":
    main()