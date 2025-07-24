#!/usr/bin/env python3
"""
Realize dialogue‑act scripts with an OpenAI model (o4‑mini by default).

Example:
python realize_openai.py \
        --scripts data/scripts.jsonl \
        --user_prompt prompts/user_prompt.txt \
        --sys_prompt prompts/sys_prompt.txt \
        --model o4-mini-2025-04-16 \
        --out_file data/dialogues.jsonl
"""
import argparse, json, os, pathlib, sys
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()  # expects OPENAI_API_KEY env var


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

def realize_one(sys_prompt, user_prompt, model):
    """Return the realized dialogue text from OpenAI."""
    resp = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        input=[{"role": "system", "content": sys_prompt},
               {"role": "user",   "content": user_prompt}],
        max_output_tokens=10_000,
    )
    return resp.output_text

def main():
    p = argparse.ArgumentParser(description="LLM realizer for DA scripts")
    p.add_argument("--scripts", type=str, required=True,
                   help="JSONL file from plan_dialogues.py")
    p.add_argument("--user_prompt", type=str, required=True,
                   help="user_prompt.txt with SCRIPT placeholders")
    p.add_argument("--sys_prompt", type=str, required=True,
                   help="sys_prompt.txt system message")
    p.add_argument("--model", type=str, default="o4-mini-2025-04-16")
    p.add_argument("--out_file", type=str, default="dialogues.jsonl")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("✖ OPENAI_API_KEY env var not set")

    sys_msg = load_text(args.sys_prompt)
    user_template = load_text(args.user_prompt)

    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.out_file, "w", encoding="utf‑8") as out_fh, \
         open(args.scripts, "r", encoding="utf‑8") as in_fh:
        for line in tqdm(in_fh):
            record = json.loads(line)
            user_msg = fill_user_template(user_template, record)
            assistant_text = realize_one(sys_msg, user_msg, args.model)
            record["messages"] = [
                {"role": "system",      "content": sys_msg},
                {"role": "user",        "content": user_msg},
                {"role": "assistant",   "content": assistant_text}
            ]
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Realized {n} dialogues → {args.out_file}")


if __name__ == "__main__":
    main()