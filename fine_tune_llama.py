#!/usr/bin/env python3
"""
Fine‑tune Llama‑3.2-3B-Instruct on synthetic dialogues produced by APIDA-Chat.

Example
-------
python fine_tune_llama.py \
    --data_file data/dialogues.jsonl \
    --output_dir outputs/allegro_lora \
    --max_steps 60
"""
import argparse, pathlib
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer


def build_dataset(path: str, tokenizer, max_seq_length: int):
    """Load JSONL with `messages` and convert to model‑ready text."""
    ds = load_dataset("json", data_files=path, split="train")
    ds = standardize_sharegpt(ds)          # normalise message dicts

    def _fmt(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return ds.map(_fmt, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True,
                    help="JSONL produced by realise_openai.py")
    ap.add_argument("--model_name",
                    default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--output_dir", default="outputs/fine_tuned")
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_rank", type=int, default=8)
    args = ap.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── load base model (4‑bit) ──────────────────────────────────────────
    max_seq_length = 20_000
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # ── attach LoRA adapter ──────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # ── chat template + dataset ──────────────────────────────────────────
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    dataset = build_dataset(args.data_file, tokenizer, max_seq_length)

    # ── training setup ───────────────────────────────────────────────────
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        warmup_steps=5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅  LoRA fine‑tuning complete → {args.output_dir}")


if __name__ == "__main__":
    main()