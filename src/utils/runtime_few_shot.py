from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

from src.agent import PromptSafetyAgent
from src.algorithms import configure_model_adapter


DEFAULT_ALGORITHM_NAME = "evaluate_rewrite"
DEFAULT_DATASET_PATH = "data/adl_final_25w_part1_with_cost.jsonl"


def parse_args() -> argparse.Namespace:
    """
    1. inputs: 無
    2. 說明: 解析 CLI 參數，包含資料集路徑、演算法名稱與 few-shot 檔案。
    3. return: argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="Run inference with per-category few-shot prompts.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset path or HF repo id (預設: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=DEFAULT_ALGORITHM_NAME,
        help="Algorithm function name defined in src/algorithms.py（預設 evaluate_rewrite）。",
    )
    parser.add_argument(
        "--few-shot",
        type=Path,
        required=True,
        help="few_shot.jsonl 路徑（category -> [{orig, rewrite}] 的單一 JSON 物件）。",
    )
    return parser.parse_args()


def _load_few_shot(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    1. inputs: path (few_shot.jsonl 的路徑)
    2. 說明: 讀取 category 對應的 few-shot 範例表，確保結構為 dict[str, list[dict]]。
    3. return: few-shot 資料字典。
    """
    if not path.is_file():
        raise FileNotFoundError(f"few-shot file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    normalized: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(data, dict):
        raise ValueError("few-shot file must contain a JSON object of category -> list.")
    for category, samples in data.items():
        if not isinstance(samples, list):
            continue
        normalized_samples: List[Dict[str, str]] = []
        for item in samples:
            if not isinstance(item, dict):
                continue
            normalized_samples.append(
                {
                    "orig": str(item.get("orig", "")),
                    "rewrite": str(item.get("rewrite", "")),
                }
            )
        normalized[category] = normalized_samples
    return normalized


def _derive_paths(dataset: str, algorithm: str) -> Tuple[Path, Path, str]:
    """
    1. inputs: dataset (資料集路徑或 HF id), algorithm (演算法名稱)
    2. 說明: 建立輸出資料夾與 prompts 檔案路徑。
    3. return: (output_dir, prompts_file, dataset_name)。
    """
    dataset_name = dataset.split("/")[-1].split(".")[0]
    output_dir = Path(f"results/{algorithm}")
    prompts_file = output_dir / f"prompts_{dataset_name}.jsonl"
    return output_dir, prompts_file, dataset_name


def _load_dataset(dataset_path: str) -> Tuple[Dataset, str]:
    """
    1. inputs: dataset_path (jsonl 路徑或 HF repo id)
    2. 說明: 載入資料集並回傳 Dataset 與 split 名稱；要求至少包含 prompt 欄位，若有 category 會用於 few-shot。
    3. return: (Dataset, split_name)。
    """
    if os.path.isfile(dataset_path):
        file_extension = dataset_path.split(".")[-1]
        if file_extension == "jsonl":
            dataset_dict = load_dataset("json", data_files=dataset_path)
        else:
            raise ValueError(f"Unsupported single file type: {file_extension}. Must be .jsonl or a directory/Hub ID.")
    elif os.path.exists(dataset_path):
        dataset_dict = load_dataset(dataset_path)
    else:
        dataset_dict = load_dataset(dataset_path)

    split_name = list(dataset_dict.keys())[0]
    ds: Dataset = dataset_dict[split_name]
    if "prompt" not in ds.column_names:
        raise ValueError(f"Dataset split '{split_name}' must contain a 'prompt' field.")
    return ds, split_name


def main() -> None:
    """
    1. inputs: 無
    2. 說明: 讀取 few-shot 檔案，依 category 拼接 few-shot system prompt 執行推論並寫出 prompts JSONL。
    3. return: None
    """
    args = parse_args()
    few_shot_data = _load_few_shot(args.few_shot)

    output_dir, prompts_file, dataset_name = _derive_paths(args.dataset, args.algorithm)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Few-shot Inference ===")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Few-shot file: {args.few_shot}")
    print(f"Output file: {prompts_file}")

    # 啟用 few-shot
    configure_model_adapter(few_shot_enabled=True, few_shot_examples=few_shot_data)
    agent = PromptSafetyAgent(args.algorithm)

    ds, split_name = _load_dataset(args.dataset)
    total = len(ds)

    start_index = 0
    if prompts_file.exists():
        try:
            with prompts_file.open("r", encoding="utf-8") as f:
                start_index = len(f.readlines())
            print(f"Resume from line {start_index} (existing {prompts_file}).")
        except Exception as exc:
            print(f"Warning: failed to read existing prompts file, will overwrite. Reason: {exc}")
            start_index = 0

    mode = "a" if start_index else "w"
    with prompts_file.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(ds):
            if index < start_index:
                continue
            toxic_prompt = record["prompt"]
            category = record.get("category")
            rewritten_prompt = agent.rewrite(toxic_prompt, category=category)
            handle.write(json.dumps(rewritten_prompt, ensure_ascii=False) + "\n")
            if (index + 1) % 10 == 0 or index == total - 1:
                print(f"Processed {index + 1} / {total} prompts in split '{split_name}'.")

    print(f"Done. Few-shot prompts saved to: {prompts_file}")


if __name__ == "__main__":
    main()
