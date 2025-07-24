#!/usr/bin/env python3
"""
Generate dialogue‑act scripts with the legacy dm4api planner.

Example:
python plan_dialogues.py \
        --n_scripts 250 \
        --api_dir dm4api/data/apis/allegro \
        --model_dir dm4api/data/weights/VanillaDense_DefaultEnv_allegro/10000 \
        --schema_dir dm4api/data/schema/new_scheme \
        --out_file data/scripts.jsonl
"""
import argparse, json, os, sys, pathlib
from tqdm import tqdm

# ── import dm4api ─────────────────────────
dm4apidir = os.path.join("dm4api", "src")
sys.path.append(dm4apidir)
from dm4api import create_env, create_agent                                    # noqa: E402

# ──────────────────────────────────────────────────────────────────────────


def run_episode(env, agent):
    """Return a JSON‑serialisable dict with the DA script and metadata."""
    agent.reset_states()
    env.interactive_reset()

    script_lines, da_list = [], []

    # first USER turn ------------------------------------------------------
    turn_id = 1
    observation = env.user_step()
    user_da = env.last_user_action
    script_lines.append(f"{turn_id}\tuser\t{user_da}")
    da_list.append(user_da)

    # iterate dialogue until terminal -------------------------------------
    while not env.isDone():
        turn_id += 1
        raw_sys_da = agent.forward(observation)
        sys_da = env.interactive_system_step(raw_sys_da)
        script_lines.append(f"{turn_id}\tsystem\t{sys_da}")
        da_list.append(sys_da)

        turn_id += 1
        observation = env.user_step()
        user_da = env.last_user_action
        script_lines.append(f"{turn_id}\tuser\t{user_da}")
        da_list.append(user_da)

    return {
        "target_function": env.user.constraints["target_function_named"],
        "da_script": "\n".join(script_lines),
        "da_list": [str(da) for da in da_list],  # stringify for JSONL
    }


def main():
    p = argparse.ArgumentParser(description="Generate dialogue‑act scripts")
    p.add_argument("--n_scripts", type=int, default=250)
    p.add_argument("--api_dir", type=str, required=True,
                   help="directory with the target API docs")
    p.add_argument("--model_dir", type=str, required=True,
                   help="path to the trained DM weights")
    p.add_argument("--schema_dir", type=str, required=True)
    p.add_argument("--out_file", type=str, default="data/scripts.jsonl")
    args = p.parse_args()

    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    env = create_env("DefaultEnv",
                     schemedir=args.schema_dir,
                     apidir=args.api_dir)
    agent = create_agent("DefaultLearnedAgent", 
                     schemedir=args.schema_dir,
                     modeldir=args.model_dir)

    with open(args.out_file, "w", encoding="utf‑8") as fh:
        for _ in tqdm(range(args.n_scripts)):
            episode = run_episode(env, agent)
            fh.write(json.dumps(episode, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n_scripts} scripts → {args.out_file}")


if __name__ == "__main__":
    main()