from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


class FewShotSelector:
    def __init__(
        self,
        trial_dirs: List[Path],
        dataset_path: Path,
        num_per_category: int,
        target_safety: float = 0.5,
        target_relevance: float = 1.0,
        output_path: Optional[Path] = None,
        raw_filename: str = "raw_adl_final_25w_part1_with_cost.jsonl",
    ):
        """
        1. inputs: trial_dirs (一或多個試驗資料夾), dataset_path (原始資料集 jsonl), num_per_category (每個類別取樣數), target_safety, target_relevance, output_path, raw_filename
        2. 說明: 設定 few-shot 抽樣所需的路徑、篩選條件與每類輸出數量。
        3. return: None
        """
        self.trial_dirs = trial_dirs
        self.dataset_path = dataset_path
        self.num_per_category = num_per_category
        self.target_safety = target_safety
        self.target_relevance = target_relevance
        self.raw_filename = raw_filename
        self.output_path = output_path or Path("data/few_shot/few_shot.jsonl")
        self.dataset_maps: Dict[str, Dict[int, Dict[str, object]]] = {}
        self.dataset_dir = self.dataset_path.parent if self.dataset_path.parent != Path("") else Path(".")

    def _load_dataset_map_from_path(self, dataset_path: Path) -> Dict[int, Dict[str, object]]:
        """
        1. inputs: dataset_path（要讀取的資料集 jsonl 檔案路徑）
        2. 說明: 讀取指定的資料集，建立 question_id 對應 category 與 cost 的查詢表。
        3. return: dict，key 為 question_id，value 包含 category 與 cost。
        """
        dataset_map: Dict[int, Dict[str, object]] = {}
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                question_id = int(record["question_id"])
                dataset_map[question_id] = {
                    "category": record.get("category", ""),
                    "cost": record.get("cost"),
                }
        return dataset_map

    def _resolve_dataset_path(self, dataset_name: str) -> Path:
        """
        1. inputs: dataset_name（從 raw 檔名稱分析出的資料集識別字串）
        2. 說明: 根據 dataset_name 先嘗試使用指定 dataset_path，再在預設資料夾中尋找對應 jsonl。
        3. return: 對應的 dataset_path。
        """
        dataset_filename = f"{dataset_name}.jsonl"
        normalized_filename = dataset_filename.casefold()
        if self.dataset_path.name.casefold() == normalized_filename and self.dataset_path.is_file():
            return self.dataset_path
        fallback_path = self.dataset_dir / dataset_filename
        if fallback_path.is_file():
            return fallback_path
        for candidate in self.dataset_dir.glob("*.jsonl"):
            if candidate.name.casefold() == normalized_filename:
                return candidate
        raise FileNotFoundError(
            f"dataset file for '{dataset_name}' not found (tried {self.dataset_path} and {fallback_path})"
        )

    def _get_dataset_map(self, dataset_name: str) -> Dict[int, Dict[str, object]]:
        """
        1. inputs: dataset_name（raw 檔名稱蘊含的資料集身分）
        2. 說明: 快取各 dataset_name 的 question_id 查詢表，必要時重新讀取。
        3. return: dataset_name 對應的查詢表。
        """
        if dataset_name not in self.dataset_maps:
            dataset_path = self._resolve_dataset_path(dataset_name)
            self.dataset_maps[dataset_name] = self._load_dataset_map_from_path(dataset_path)
        return self.dataset_maps[dataset_name]

    def _dataset_name_from_raw_path(self, raw_path: Path) -> str:
        """
        1. inputs: raw_path（raw 檔案路徑）
        2. 說明: 解析 raw 檔名稱以取得資料集的識別字串（去除 raw_ 前綴）。
        3. return: 資料集名稱（例：adl_final_25w_part2_with_cost）。
        """
        stem = raw_path.stem
        prefix = "raw_"
        if stem.startswith(prefix):
            return stem[len(prefix) :]
        return stem

    def _discover_raw_paths(self, trial_dir: Path) -> List[Path]:
        """
        1. inputs: trial_dir（trial 目錄）
        2. 說明: 優先使用指定 raw_filename，其次尋找試驗資料夾中的 raw_*.jsonl 檔案。
        3. return: 符合條件的 raw 欄位路徑列表。
        """
        results: List[Path] = []
        if self.raw_filename:
            candidate = trial_dir / self.raw_filename
            if candidate.is_file():
                return [candidate]
        results.extend(sorted(trial_dir.glob("raw_*.jsonl")))
        return results

    def _iter_raw_records(self) -> Iterable[tuple[Dict[str, object], str]]:
        """
        1. inputs: 無
        2. 說明: 依序讀取各 trial 資料夾中的 raw 檔案，產生 raw 記錄與資料集名稱。
        3. return: 產生器，每次回傳 (record, dataset_name)。
        """
        for trial_dir in self.trial_dirs:
            raw_paths = self._discover_raw_paths(trial_dir)
            if not raw_paths:
                print(f"Warning: raw file not found, skip: {trial_dir}")
                continue
            for raw_path in raw_paths:
                dataset_name = self._dataset_name_from_raw_path(raw_path)
                with raw_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        yield json.loads(line), dataset_name

    def _score_bucket(self, safety: Optional[float], relevance: Optional[float]) -> Optional[str]:
        """
        1. inputs: safety, relevance（待檢查的分數）
        2. 說明: 依分數決定候選層級：優先 1.0/1.0，不足再用 0.5/1.0。
        3. return: "primary" | "fallback" | None。
        """
        if safety is None or relevance is None:
            return None
        safety_f = float(safety)
        relevance_f = float(relevance)
        if abs(safety_f - 1.0) < 1e-6 and abs(relevance_f - 1.0) < 1e-6:
            return "primary"
        if abs(safety_f - 0.5) < 1e-6 and abs(relevance_f - 1.0) < 1e-6:
            return "fallback"
        return None

    def _build_candidates(self) -> DefaultDict[str, Dict[str, List[Dict[str, object]]]]:
        """
        1. inputs: 無
        2. 說明: 篩選符合分數條件的 raw 紀錄，依據 dataset 屬性找出對應 metadata，再依 category 分 primary/fallback 分組。
        3. return: 預設 dict，key 為 category，value 為 {"primary": [...], "fallback": [...]}。
        """
        candidates: DefaultDict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(
            lambda: {"primary": [], "fallback": []}
        )

        for record, dataset_name in self._iter_raw_records():
            dataset_map = self._get_dataset_map(dataset_name)
            raw_id = int(record.get("id"))
            question_id = raw_id + 1
            meta = dataset_map.get(question_id)
            if not meta:
                print(f"Warning: question_id {question_id} not found in dataset; skip.")
                continue

            bucket = self._score_bucket(record.get("safety_score"), record.get("relevance_score"))
            if bucket is None:
                continue

            category = str(meta.get("category", ""))
            cost = meta.get("cost", record.get("cost"))
            candidates[category][bucket].append(
                {
                    "category": category,
                    "id": question_id,
                    "original_prompt": record.get("original_prompt", ""),
                    "rewritten_prompt": record.get("rewritten_prompt", ""),
                    "safety_score": record.get("safety_score"),
                    "relevance_score": record.get("relevance_score"),
                    "cost": cost,
                }
            )

        return candidates

    def _select_rows(
        self, candidates: DefaultDict[str, Dict[str, List[Dict[str, object]]]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        1. inputs: candidates（依 category 分組的候選列表）
        2. 說明: 依 category 排序並各自優先取 primary，不足再補 fallback，總數為 num_per_category，並轉成 {orig, rewrite}。
        3. return: dict[category] -> list[{"orig","rewrite"}]。
        """
        rows: Dict[str, List[Dict[str, str]]] = {}
        for category in sorted(candidates.keys()):
            primary = sorted(candidates[category]["primary"], key=lambda item: item["id"])
            fallback = sorted(candidates[category]["fallback"], key=lambda item: item["id"])

            combined: List[Dict[str, object]] = []
            combined.extend(primary)
            if len(combined) < self.num_per_category:
                combined.extend(fallback[: self.num_per_category - len(combined)])

            if len(combined) < self.num_per_category:
                print(
                    f"Warning: category '{category}' has only {len(combined)} matches (< {self.num_per_category})."
                )
            rows[category] = [
                {
                    "orig": rec.get("original_prompt", ""),
                    "rewrite": rec.get("rewritten_prompt", ""),
                }
                for rec in combined[: self.num_per_category]
            ]
        return rows

    def _write_output(self, rows: Dict[str, List[Dict[str, str]]]) -> Path:
        """
        1. inputs: rows（category -> few-shot 範例列表）
        2. 說明: 將 rows 以單一 JSON 物件寫入指定的輸出路徑。
        3. return: 寫出的檔案路徑。
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(rows, ensure_ascii=False, indent=2))
        return self.output_path

    def run(self) -> Path:
        """
        1. inputs: 無
        2. 說明: 讀取資料集與所有 raw 檔，篩選符合條件的紀錄並輸出 few_shot.jsonl。
        3. return: few_shot.jsonl 的路徑。
        """
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"dataset file not found: {self.dataset_path}")

        candidates = self._build_candidates()
        rows = self._select_rows(candidates)
        output_path = self._write_output(rows)
        print(f"few_shot.jsonl written to: {output_path}")
        return output_path


def parse_args() -> argparse.Namespace:
    """
    1. inputs: 無
    2. 說明: 解析 CLI 參數，包含一個或多個 trial 目錄、資料集路徑與每類取樣數。
    3. return: argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="Select few-shot samples from a trial folder.")
    parser.add_argument(
        "trial_dirs",
        nargs="+",
        type=Path,
        help="一個或多個 trial 資料夾 (e.g., results/part1/evaluate_rewrite_42_0.570)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/adl_final_25w_part1_with_cost.jsonl"),
        help="Path to the original dataset jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/few_shot/few_shot.jsonl"),
        help="合併輸出的 few_shot.jsonl 路徑。",
    )
    parser.add_argument(
        "-n",
        "--num-per-category",
        type=int,
        default=3,
        help="Number of samples per category.",
    )
    return parser.parse_args()


def main() -> None:
    """
    1. inputs: 無
    2. 說明: 依據 CLI 參數執行 few-shot 選擇與輸出，支援一次處理多個 trial 資料夾並合併到單一路徑。
    3. return: None
    """
    args = parse_args()
    selector = FewShotSelector(
        trial_dirs=args.trial_dirs,
        dataset_path=args.dataset,
        num_per_category=args.num_per_category,
        output_path=args.output,
    )
    selector.run()


if __name__ == "__main__":
    main()

"""
python src/utils/select_few_shots.py results/part1/evaluate_rewrite_42_0.570 -n 5

python src/utils/select_few_shots.py \
results/part1/evaluate_rewrite_21_0.515 \
  results/part1/evaluate_rewrite_42_0.570 \
  -n 9 \
  --dataset data/adl_final_25w_part1_with_cost.jsonl \
  --output data/few_shot/merged_few_shot_n_9.jsonl
"""
