#!/usr/bin/env python3
"""
Realise dialogue‑act scripts with a local Llama‑3.1 8B model.

Example
python realise_llama.py \
        --scripts data/scripts.jsonl \
        --sys_prompt prompts/sys_prompt.txt \
        --user_prompt prompts/user_prompt.txt \
        --model outputs/fine_tune  \
        --out_file data/dialogues_llama.jsonl
"""
import unsloth
import argparse, json, os, pathlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

DELIM = "Produce the JSON now.assistant"  # string before generated text


def load_text(path):
    with open(path, "r", encoding="utf‑8") as fh:
        return fh.read()

def fill_user_template(user_template, script_record):
    target_fn = script_record["target_function"]
    da_script = script_record["da_script"]
    user_prompt = user_template.replace("[REVEAL TARGET FUNCTION]",
                                f"Target function: {target_fn}\n")
    user_prompt = user_prompt.replace("[REPLACE WITH SCRIPT]", da_script)
    return user_prompt

def realise_one(model, tokenizer, sys_msg, user_msg, max_new_tokens=4096):
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize = True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache = True,
        temperature = .5,
        min_p = .01,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt echo
    try:
        generated = full_text.split(DELIM, 1)[1].strip()
    except IndexError:
        generated = full_text[-max_new_tokens:].strip()  # fallback
    return generated


def main():
    ap = argparse.ArgumentParser(description="Realise DA scripts with local Llama")
    ap.add_argument("--scripts", required=True, help="JSONL produced by plan_dialogues.py")
    ap.add_argument("--sys_prompt", required=True, help="system prompt file")
    ap.add_argument("--user_prompt", required=True, help="user prompt template")
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                    help="HF hub ID or local dir (fine‑tuned)")
    ap.add_argument("--out_file", default="dialogues_llama.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys_msg = load_text(args.sys_prompt)
    user_template = load_text(args.user_prompt)

    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    max_seq_length = 20_000
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype = None,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    n = 0
    with open(args.out_file, "w", encoding="utf‑8") as out_fh, \
         open(args.scripts, "r", encoding="utf‑8") as in_fh:
        for line in tqdm(in_fh):
            record = json.loads(line)
            user_msg = fill_user_template(user_template, record)
            assistant_text = realise_one(model, tokenizer, sys_msg, user_msg, max_new_tokens=10000)
            record["messages"] = [
                {"role": "system",    "content": sys_msg},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_text},
            ]
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Realised {n} dialogues → {args.out_file}")


if __name__ == "__main__":
    main()