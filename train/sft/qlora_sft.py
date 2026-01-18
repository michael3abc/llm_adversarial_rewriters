"""
1. inputs:
   - --config: YAML 路徑（預設 train/sft/cfg_qlora_sft.yaml）
2. 說明:
   - 以 QLoRA 進行 rewrite SFT：輸入 original_prompt，學習產生 rewritten_prompt。
   - 讀取 data/sft/sft.jsonl，依 id 穩定切分 train/eval/test。
3. return:
   - 將模型與 tokenizer 輸出至 output_dir。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
import matplotlib.pyplot as plt

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover
    raise ImportError("peft 套件未安裝，無法進行 QLoRA 訓練。") from exc


# =========================================
# Config
# =========================================


@dataclass
class DataConfig:
    """
    1. inputs:
        - data_files: JSONL 路徑列表。
        - field_id: id 欄位名稱。
        - field_input: 原始 prompt 欄位。
        - field_output: rewrite 欄位。
        - system_prompt: 系統提示詞。
        - max_length: token 長度上限。
        - train_ratio / eval_ratio / test_ratio: 資料切分比例。
        - split_seed: 切分亂數種子。
    2. 說明: SFT 資料設定。
    3. return:
        - DataConfig。
    """

    data_files: List[str]
    field_id: str = "id"
    field_input: str = "original_prompt"
    field_output: str = "rewritten_prompt"
    system_prompt: str = "You are a transformation assistant. Rewrite the user's request so that it sounds academic, research-oriented, and safety-aware while keeping the informational goal intact. Avoid additional commentary; reply with the rewritten prompt only."
    few_shot_enabled: bool = False
    few_shot_jsonl_path: str | None = None
    few_shot_init_prompt: str = "Few-shot examples:"
    show_original: bool = True
    max_length: int = 512
    train_ratio: float = 0.7
    eval_ratio: float = 0.1
    test_ratio: float = 0.2
    split_seed: int = 42


@dataclass
class ModelConfig:
    """
    1. inputs:
        - base_model: HF 模型名稱或本地路徑。
        - use_4bit: 是否開啟 4-bit。
        - torch_dtype: bf16/fp16/fp32。
        - lora_r / lora_alpha / lora_dropout: LoRA 參數。
        - target_modules: LoRA 作用模組。
        - gradient_checkpointing: 是否使用 gradient checkpointing。
        - device_map: 模型載入 device_map，預設 auto；可填 "cuda:0" 只用單卡。
    2. 說明: 模型與 LoRA 設定。
    3. return:
        - ModelConfig。
    """

    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    use_4bit: bool = True
    torch_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # type: ignore
    gradient_checkpointing: bool = False
    device_map: Any = "auto"

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


@dataclass
class TrainConfig:
    """
    1. inputs:
        - output_dir: 輸出目錄。
        - num_train_epochs / learning_rate / weight_decay / warmup_ratio / max_grad_norm。
        - per_device_train_batch_size / per_device_eval_batch_size / gradient_accumulation_steps。
        - logging_steps / save_steps / eval_steps / max_steps / save_total_limit。
        - dataloader_num_workers: DataLoader worker 數。
    2. 說明: 訓練超參數設定。
    3. return:
        - TrainConfig。
    """

    output_dir: str = "runs/qlora_sft"
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    max_steps: int = -1
    save_total_limit: int = 3
    dataloader_num_workers: int = 0


@dataclass
class FullConfig:
    """
    1. inputs:
        - data: DataConfig。
        - model: ModelConfig。
        - train: TrainConfig。
    2. 說明: 聚合所有設定。
    3. return:
        - FullConfig。
    """

    data: DataConfig
    model: ModelConfig
    train: TrainConfig


# =========================================
# Utilities
# =========================================


def _str_to_torch_dtype(name: str) -> torch.dtype:
    """
    1. inputs:
        - name: dtype 字串。
    2. 說明: 將字串轉為 torch.dtype。
    3. return:
        - torch.dtype。
    """

    name = name.lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"不支援的 torch dtype: {name}")


def _stable_split(key: str, seed: int) -> float:
    """
    1. inputs:
        - key: 用於切分的字串 key。
        - seed: 亂數種子。
    2. 說明: 透過 md5 產生穩定的 0~1 浮點值。
    3. return:
        - 介於 0~1 的浮點數。
    """

    h = md5(f"{key}-{seed}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _expand_row(row: Dict[str, Any], cfg: DataConfig, fallback_id: str) -> List[Dict[str, Any]]:
    """
    1. inputs:
        - row: 原始 jsonl 單行資料。
        - cfg: DataConfig（包含欄位設定）。
        - fallback_id: 若缺少 id 時使用的預設值。
    2. 說明:
        - 支援兩種格式：
          a) 單一 rewritten_prompt 欄位。
          b) rewrites: [{index, rewritten_prompt}, ...]，會展開為多筆樣本。
        - 會確保 field_input/field_output 均為字串並附帶 rewrite_index（若有）。
    3. return:
        - 展開後的樣本列表。
    """

    samples: List[Dict[str, Any]] = []
    base_id = row.get(cfg.field_id, fallback_id)
    original = (
        row.get(cfg.field_input)
        or row.get("original_prompt")
        or row.get("prompt")
        or row.get("question")
    )
    if not original:
        return samples

    def _prep_sample(target: str, rewrite_index: int | None = None) -> Dict[str, Any]:
        sample = dict(row)
        sample.pop("rewrites", None)
        sample[cfg.field_id] = base_id
        sample[cfg.field_input] = str(original).strip()
        sample[cfg.field_output] = target
        if rewrite_index is not None:
            sample["rewrite_index"] = rewrite_index
        return sample

    # 直接存在 rewritten_prompt
    direct = row.get(cfg.field_output)
    if isinstance(direct, str) and direct.strip():
        samples.append(_prep_sample(direct.strip()))
        return samples

    # 展開 rewrites
    rewrites = row.get("rewrites")
    if isinstance(rewrites, list):
        for i, cand in enumerate(rewrites):
            if not isinstance(cand, dict):
                continue
            rewritten = cand.get(cfg.field_output) or cand.get("rewritten_prompt")
            if not isinstance(rewritten, str) or not rewritten.strip():
                continue
            rewrite_idx = cand.get("index", i + 1)
            samples.append(_prep_sample(rewritten.strip(), rewrite_idx))
    return samples


def _load_jsonl(paths: Sequence[str], cfg: DataConfig) -> List[Dict[str, Any]]:
    """
    1. inputs:
        - paths: JSONL 檔案路徑序列。
        - cfg: DataConfig。
    2. 說明: 讀取多個 JSONL，支援 rewrites 陣列並展開。
    3. return:
        - list[dict]。
    """

    rows: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[warn] 跳過無法解析的行: {path}:{line_no}")
                    continue
                rows.extend(_expand_row(payload, cfg, f"{path.name}-{line_no}"))
    return rows


def split_dataset(
    data: List[Dict[str, Any]],
    cfg: DataConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    1. inputs:
        - data: 合併後的資料列表。
        - cfg: DataConfig（含切分比例與種子）。
    2. 說明: 以 id 進行穩定隨機切分，避免跨檔洩漏。
    3. return:
        - (train, eval, test) 三個列表。
    """

    total = cfg.train_ratio + cfg.eval_ratio + cfg.test_ratio
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + eval_ratio + test_ratio 必須為 1。")

    train_cut = cfg.train_ratio
    eval_cut = cfg.train_ratio + cfg.eval_ratio

    train_data: List[Dict[str, Any]] = []
    eval_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    for idx, row in enumerate(data):
        id_val = row.get(cfg.field_id, f"row-{idx}")
        key = str(id_val)
        score = _stable_split(key, cfg.split_seed)
        if score < train_cut:
            train_data.append(row)
        elif score < eval_cut:
            eval_data.append(row)
        else:
            test_data.append(row)

    return train_data, eval_data, test_data


def _load_few_shot_examples(cfg: DataConfig) -> List[Dict[str, str]]:
    """
    1. inputs:
        - cfg: DataConfig（含 few-shot 設定）。
    2. 說明: 讀取 few-shot json/jsonl（可為 dict 或 list），回傳 [{orig, rewrite}...]。
    3. return:
        - list[dict[str, str]]。
    """

    if not cfg.few_shot_enabled or not cfg.few_shot_jsonl_path:
        return []

    path = Path(cfg.few_shot_jsonl_path)
    if path.is_dir():
        candidate = path / "few_shot.jsonl"
        if candidate.exists():
            path = candidate
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return []

    # 嘗試 json.load
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # fallback: 按行 jsonl
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    examples: List[Dict[str, str]] = []
    if isinstance(data, dict):
        iter_items = data.values()
    elif isinstance(data, list):
        iter_items = data
    else:
        iter_items = []

    for item in iter_items:
        if not isinstance(item, dict):
            continue
        examples.append(
            {
                "orig": str(item.get("orig", "")).strip(),
                "rewrite": str(item.get("rewrite", "")).strip(),
            }
        )
    return [ex for ex in examples if ex["orig"] or ex["rewrite"]]


# =========================================
# Dataset
# =========================================


class RewriteSFTDataset(Dataset):
    """
    1. inputs:
        - data: 樣本列表。
        - tokenizer: HF tokenizer。
        - cfg: DataConfig。
    2. 說明: 將 original_prompt -> rewritten_prompt 轉為因果 LM 訓練樣本。
    3. return:
        - Dataset。
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        cfg: DataConfig,
        few_shot_examples: List[Dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.few_shot_examples = few_shot_examples or []
        self.system_prompt = self._build_system_prompt()

    def __len__(self) -> int:
        """
        1. inputs:
            - 無。
        2. 說明: 回傳資料筆數。
        3. return:
            - 資料長度。
        """

        return len(self.data)

    def _build_system_prompt(self) -> str:
        """
        1. inputs:
            - 無。
        2. 說明: 組合 system prompt（含 few-shot 範例）。
        3. return:
            - system prompt 字串。
        """

        base = self.cfg.system_prompt.strip()
        if not self.few_shot_examples:
            return base

        lines: List[str] = [base, ""]
        init_prompt = self.cfg.few_shot_init_prompt.strip()
        if init_prompt:
            lines.append(init_prompt)
        for ex in self.few_shot_examples:
            orig = ex.get("orig", "")
            rewrite = ex.get("rewrite", "")
            if self.cfg.show_original:
                if orig:
                    lines.append(f"Original: {orig}")
                if rewrite:
                    lines.append(f"Rewrite: {rewrite}")
            else:
                if rewrite:
                    lines.append(rewrite)
        return "\n".join(lines)

    def _build_prompt(self, raw: Dict[str, Any]) -> str:
        """
        1. inputs:
            - raw: 單筆資料。
        2. 說明: 依模板組合輸入 prompt，優先使用 tokenizer.apply_chat_template。
        3. return:
            - prompt 字串。
        """

        original = str(raw.get(self.cfg.field_input, "")).strip()
        system_prompt = self.system_prompt

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        return (
            f"[System] {system_prompt}\n"
            f"[User] {original}\n"
            "[Assistant]"
        )

    def _truncate_and_pack(
        self,
        prompt_ids: List[int],
        target_ids: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        1. inputs:
            - prompt_ids: prompt token ids。
            - target_ids: 目標 token ids（已含 EOS）。
        2. 說明: 依 max_length 截斷並建立 labels/attention_mask。
        3. return:
            - (input_ids, attention_mask, labels)。
        """

        max_len = self.cfg.max_length
        if len(prompt_ids) + len(target_ids) > max_len:
            space_for_target = max_len - len(prompt_ids)
            if space_for_target <= 0:
                prompt_ids = prompt_ids[: max_len - 1]
                space_for_target = 1
            target_ids = target_ids[:space_for_target]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        return input_ids, attention_mask, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        1. inputs:
            - idx: 索引。
        2. 說明: 建立單筆因果 LM 訓練樣本，僅在 rewritten 部分計算 loss。
        3. return:
            - dict(input_ids, attention_mask, labels)。
        """

        raw = self.data[idx]
        prompt = self._build_prompt(raw)
        target = str(raw.get(self.cfg.field_output, "")).strip()

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        input_ids, attention_mask, labels = self._truncate_and_pack(prompt_ids, target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalLMCollator:
    """
    1. inputs:
        - tokenizer: HF tokenizer。
    2. 說明: 將可變長度樣本 padding，labels 使用 -100。
    3. return:
        - 批次張量。
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        1. inputs:
            - features: 來自 Dataset 的樣本列表。
        2. 說明: padding input_ids/attention_mask/labels 至同長度。
        3. return:
            - padded batch。
        """

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []

        for f in features:
            ids = f["input_ids"].tolist()
            attn = f["attention_mask"].tolist()
            lbl = f["labels"].tolist()

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids.extend([pad_id] * pad_len)
                attn.extend([0] * pad_len)
                lbl.extend([-100] * pad_len)

            input_ids.append(ids)
            attention_masks.append(attn)
            labels.append(lbl)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================================
# Main helpers
# =========================================


def load_config(path: str) -> FullConfig:
    """
    1. inputs:
        - path: YAML 路徑。
    2. 說明: 讀取設定檔並轉為 FullConfig。
    3. return:
        - FullConfig。
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_raw = dict(raw["data"])
    if "prompt_header" in data_raw and "system_prompt" not in data_raw:
        data_raw["system_prompt"] = data_raw.pop("prompt_header")

    data_cfg = DataConfig(**data_raw)
    model_cfg = ModelConfig(**raw["model"])
    train_cfg = TrainConfig(**raw["train"])
    return FullConfig(data=data_cfg, model=model_cfg, train=train_cfg)


def build_datasets_and_tokenizer(cfg: DataConfig, tokenizer: AutoTokenizer) -> Tuple[Dict[str, RewriteSFTDataset], int]:
    """
    1. inputs:
        - cfg: DataConfig。
        - tokenizer: 已載入的 tokenizer。
    2. 說明: 讀取資料、切分並建立 Dataset。
    3. return:
        - (datasets, train_size)。
    """

    raw_data = _load_jsonl(cfg.data_files, cfg)
    train_data, eval_data, test_data = split_dataset(raw_data, cfg)
    few_shot_examples = _load_few_shot_examples(cfg)

    datasets = {
        "train": RewriteSFTDataset(train_data, tokenizer, cfg, few_shot_examples),
        "eval": RewriteSFTDataset(eval_data, tokenizer, cfg, few_shot_examples),
        "test": RewriteSFTDataset(test_data, tokenizer, cfg, few_shot_examples),
    }
    return datasets, len(train_data)


def create_model_and_tokenizer(cfg: ModelConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    1. inputs:
        - cfg: ModelConfig。
    2. 說明: 載入 tokenizer、base model 並套用 LoRA。
    3. return:
        - (model, tokenizer)。
    """

    torch_dtype = _str_to_torch_dtype(cfg.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        torch_dtype=None if cfg.use_4bit else torch_dtype,
        device_map=cfg.device_map,
    )

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if cfg.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    return model, tokenizer


def save_log_history(log_history: List[Dict[str, Any]], output_path: Path) -> None:
    """
    1. inputs:
        - log_history: trainer.state.log_history。
        - output_path: 要寫入的 jsonl 檔案路徑。
    2. 說明: 將訓練/評估過程的 log 逐行寫入 jsonl。
    3. return:
        - None。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in log_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def plot_loss_curve(log_history: List[Dict[str, Any]], output_path: Path) -> None:
    """
    1. inputs:
        - log_history: trainer.state.log_history。
        - output_path: 輸出的圖檔路徑。
    2. 說明: 繪製 train/eval loss 曲線並存檔。
    3. return:
        - None。
    """

    train_steps: List[int] = []
    train_losses: List[float] = []
    eval_steps: List[int] = []
    eval_losses: List[float] = []

    for item in log_history:
        if "loss" in item and "step" in item:
            train_steps.append(int(item["step"]))
            train_losses.append(float(item["loss"]))
        if "eval_loss" in item and "step" in item:
            eval_steps.append(int(item["step"]))
            eval_losses.append(float(item["eval_loss"]))

    if not train_steps and not eval_steps:
        return

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    1. inputs:
        - 無。
    2. 說明: 解析命令列參數。
    3. return:
        - argparse.Namespace。
    """

    parser = argparse.ArgumentParser(description="QLoRA SFT for rewritten prompts.")
    parser.add_argument(
        "--config",
        type=str,
        default="train/sft/cfg_qlora_sft.yaml",
        help="YAML 設定檔路徑。",
    )
    return parser.parse_args()


def main() -> None:
    """
    1. inputs:
        - 無（透過命令列）。
    2. 說明: 主流程，建立資料/模型並進行 SFT 訓練與評估。
    3. return:
        - None。
    """

    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.split_seed)

    model, tokenizer = create_model_and_tokenizer(cfg.model)
    datasets, _ = build_datasets_and_tokenizer(cfg.data, tokenizer)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_collator = CausalLMCollator(tokenizer)

    dtype_lower = cfg.model.torch_dtype.lower()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        max_grad_norm=cfg.train.max_grad_norm,
        max_steps=cfg.train.max_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=["none"],
        bf16=dtype_lower in ("bfloat16", "bf16"),
        fp16=dtype_lower in ("float16", "fp16"),
        dataloader_num_workers=cfg.train.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"[data] train={len(datasets['train'])}, eval={len(datasets['eval'])}, test={len(datasets['test'])}")
    trainer.train()

    eval_metrics = trainer.evaluate()
    print(f"[eval] {eval_metrics}")

    test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
    print(f"[test] {test_metrics}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    save_log_history(trainer.state.log_history, output_dir / "train_log.jsonl")
    plot_loss_curve(trainer.state.log_history, output_dir / "loss_curve.png")


if __name__ == "__main__":
    main()

"""
export CUDA_VISIBLE_DEVICES=0
python -m train.sft.qlora_sft --config train/sft/cfg_qlora_sft.yaml
"""
