# **How to Use**
## 1. Common Data Formats (jsonl)
- Original harmful data (example: `data/adl_final_25w_part1_with_cost.jsonl`)
  ```json
  {"question_id":1,"category":"cybercrime_intrusion","prompt":"...","cost":3,"reasoning":"...","source":"ADL_Final_25W_part1"}
  ```
- SFT/pos-neg samples (`*sft*.jsonl` or `pos/neg_sample.jsonl`):
  - `{"id":1,"original_prompt":"...","rewrites":[{"index":1,"rewritten_prompt":"..."}]}`
- DPO data (`data/dpo/dpo_train_data.jsonl`):
  - `{"id":1,"original_prompt":"...","chosen":"(safe rewrite)","rejected":"(worse rewrite)"}`

## 2. sft (Supervised Rewriting)
- Data preparation:
  - `make_sft_data.py`: scan `results/**/raw_*.jsonl`, pick rewrites with safety/relevance=1.0, optional `--similarity-threshold` to dedupe, output distill-style SFT file.
    - Example: `python train/sft/make_sft_data.py --results-dir results/part1 --target-name raw_adl_final_25w_part1_with_cost.jsonl --output data/sft/sft_filt.jsonl --similarity-threshold 0.95`
  - `make_sft_data_cos.py`: similar but also outputs `pos_sample.jsonl` and `neg_sample.jsonl` for DPO, supports embedding similarity dedupe (`--similarity-threshold`, `--model-name`).
  - `r1_self_augment.py`: augment existing rewrites with chain-of-thought and output `{"prompt": "...", "completion": "<think>...</think>\n<rewrite>..."}` jsonl for R1-style SFT.
    - Example: `python -m train.sft.r1_self_augment --input data/sft/pos_sample.jsonl --output data/sft/r1_self_train.jsonl --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- Training:
  - `qlora_sft.py` + `cfg_qlora_sft.yaml`: QLoRA training for `original_prompt -> rewritten_prompt`. Each data row can be a single `rewritten_prompt` or a `rewrites` list (auto-expanded with `rewrite_index`). Few-shot is supported via `few_shot_enabled` + `few_shot_jsonl_path`.
    - Key config:
      - data: `data_files`, field names, `max_length`, train/eval/test split.
      - model: `base_model`, `use_4bit`, LoRA params, `device_map`.
      - train: epochs, LR, batch size, accumulation, log/save/eval steps.
    - Run: `python -m train.sft.qlora_sft --config train/sft/cfg_qlora_sft.yaml` (outputs to `train.train.output_dir`).

## 3. dpo
- Data source: `pos_sample.jsonl` / `neg_sample.jsonl` from `make_sft_data_cos.py` (fields: `id`, `original_prompt`, `rewrites`).
- Pairing: `make_dpo_data.py` pairs the first positive rewrite for each id with all negative rewrites, and outputs `data/dpo/dpo_train_data.jsonl`.
  - Example: `python train/dpo/make_dpo_data.py --pos data/sft/pos_sample.jsonl --neg data/sft/neg_sample.jsonl --output data/dpo/dpo_train_data.jsonl`
- Training: `run_dpo.py` reads `train/dpo/dpo.yaml` (model / data / train / lora sections).
  - Data requirements: `original_prompt`, `chosen`, `rejected`; `data.system_prompt` is applied to form ChatML.
  - Run: `python train/dpo/run_dpo.py --config train/dpo/dpo.yaml` (outputs LoRA adapter to `train.output_dir`).

## 4. Typical Workflow
1) Run inference and scoring to produce `results/**/raw_*.jsonl`.
2) Use `make_sft_data*.py` to create SFT data; use `make_dpo_data.py` if DPO is needed.
3) Train `qlora_sft.py`; optionally run `run_dpo.py` for preference tuning.
4) Re-run inference and evaluation with new trained ckpts to verify improvements.
