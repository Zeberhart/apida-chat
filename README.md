# APIDA‑chat · API‑Dialogue‑Act‑Chat
Generate domain‑grounded, multi‑turn API‑help dialogues and train a compact assistant.
---
## ✨ Key features
* Planner → Realiser → Fine‑tune three‑stage pipeline (Figure 1 in the paper).
* Ships a legacy dialogue planner for instant Allegro‑5 experiments.
* One‑command scripts for each stage; works in Colab or locally.
* Produces a LoRA‑tuned 3 B Llama model that matches teacher quality at a fraction of the cost.
* Fully modular—swap in your own planner, realiser prompt, or student backbone.

# 🚀 Quick start ([Colab](https://colab.research.google.com/drive/1pD0-IA1-yNupKQ-hA68YNxw06JlysbYu?usp=sharing))
```
git clone https://github.com/<you>/APIDA-chat.git
cd APIDA-chat

# 1. plan dialogue‑act scripts
python planners/plan_dialogues.py \
       --n_scripts 250 \
       --api_dir dm4api/data/apis/allegro \
       --policy_ckpt dm4api/data/weights/VanillaDense_DefaultEnv_allegro/10000 \
       --out_file data/scripts.jsonl

# 2. realise with OpenAI teacher (needs OPENAI_API_KEY)
python realisers/realise_openai.py \
       --scripts data/scripts.jsonl \
       --sys_prompt prompts/dev_prompt.txt \
       --prompt prompts/combined_prompt.txt \
       --model o4-mini-2025-04-16 \
       --out_file data/dialogues_openai.jsonl

# 3. fine‑tune Llama‑3.2 3B
python finetune/fine_tune_llama.py \
       --data_file data/dialogues_openai.jsonl \
       --output_dir outputs/llama_finetuned

# 4. realise with the local student model
python realisers/realise_llama.py \
       --scripts data/scripts.jsonl \
       --sys_prompt prompts/dev_prompt.txt \
       --prompt prompts/combined_prompt.txt \
       --model outputs/llama_finetuned \
       --out_file data/dialogues_llama.jsonl
```

# 🗄 Repository layout
```
APIDA-chat/
│  README.md
│  requirements.txt
│
├─ planners/          # Stage 1
│   plan_dialogues.py
│   legacy_planner/   # copied from Eberhart et al. 2021
│
├─ realizers/         # Stage 2
│   realize_openai.py
│   realize_llama.py
│
├─ finetune/          # Stage 3
│   fine_tune_llama.py
│
├─ prompts/           # system + user templates
└─ data/              # generated artefacts (git‑ignored)
```

