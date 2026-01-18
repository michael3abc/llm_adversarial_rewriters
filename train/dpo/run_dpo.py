from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_config(config_path: Path) -> Dict[str, Any]:
    """inputs: config_path: Path
    說明: 讀取 YAML 設定並回傳字典。
    return: dict
    """

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_prompt(system_prompt: str, user_prompt: str) -> str:
    """inputs: system_prompt: str, user_prompt: str
    說明: 將 system prompt 動態插入，組成 ChatML 風格的 prompt。
    return: str
    """

    return (
        f"<s>[SYSTEM]\n{system_prompt.strip()}\n[/SYSTEM]\n"
        f"[USER]\n{user_prompt.strip()}\n[/USER]\n[ASSISTANT]\n"
    )


def load_dpo_dataset(data_path: Path, system_prompt: str) -> List[Dict[str, str]]:
    """inputs: data_path: Path, system_prompt: str
    說明: 讀取 DPO jsonl，依照 system prompt 組裝 prompt，輸出 chosen/rejected。
    return: list[dict[str, str]]
    """

    rows: List[Dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"跳過無法解析的行: {data_path}:{line_no}")
                continue

            original_prompt = payload.get("original_prompt")
            chosen = payload.get("chosen")
            rejected = payload.get("rejected")
            if not original_prompt or not chosen or not rejected:
                print(f"跳過缺欄位的行: {data_path}:{line_no}")
                continue

            prompt_text = format_prompt(system_prompt, original_prompt)
            rows.append(
                {
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                    "id": payload.get("id"),
                }
            )
    return rows


def to_hf_dataset(rows: List[Dict[str, str]]):
    """inputs: rows: list[dict]
    說明: 將 list 轉為 HuggingFace Dataset，便於 TRL map pipeline。
    return: datasets.Dataset
    """

    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("需要安裝 datasets 套件以使用 DPOTrainer") from exc

    return Dataset.from_list(rows)


def load_model_and_tokenizer(model_cfg: Dict[str, Any], lora_cfg: Dict[str, Any]) -> tuple:
    """inputs: model_cfg: dict, lora_cfg: dict
    說明: 載入底模（4bit），若有 SFT adapter 則 merge，再套新的 LoRA。
    return: (model, tokenizer)
    """

    bnb_cfg = None
    if model_cfg.get("load_in_4bit", True):
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)

    device_map = model_cfg.get("device") or "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        device_map=device_map,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_adapter = model_cfg.get("sft_adapter")
    if sft_adapter:
        print(f"載入並合併 SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    return model, tokenizer


def main() -> None:
    """inputs: None
    說明: 讀取 YAML 設定，載入資料與模型，使用 DPOTrainer 執行訓練。
    return: None
    """

    parser = argparse.ArgumentParser(description="Run DPO training with YAML config.")
    parser.add_argument("--config", type=Path, default=Path("train/dpo/dpo.yaml"), help="YAML config path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    lora_cfg = cfg["lora"]

    rows = load_dpo_dataset(Path(data_cfg["train_path"]), data_cfg["system_prompt"])
    dataset = to_hf_dataset(rows)
    model, tokenizer = load_model_and_tokenizer(model_cfg, lora_cfg)

    dpo_args = DPOConfig(
        output_dir=train_cfg["output_dir"],
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        max_steps=-1,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        beta=float(train_cfg["beta"]),
        max_length=int(train_cfg["max_length"]),
        max_prompt_length=int(train_cfg["max_prompt_length"]),
    )

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }

    trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


if __name__ == "__main__":
    main()

"""
python train/dpo/run_dpo.py --config train/dpo/dpo.yaml
"""
