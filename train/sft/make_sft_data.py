from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


class RewriteSimilarityFilter:
    """inputs: threshold: float
    說明: 以餘弦相似度過濾相似 rewrite，保留較短者。
    return: None
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @staticmethod
    def _tokenize(text: str) -> Counter:
        """inputs: text: str
        說明: 簡單以單字斷詞並統計詞頻。
        return: Counter
        """

        tokens = re.findall(r"\w+", text.lower())
        return Counter(tokens)

    @staticmethod
    def _cosine(vec_a: Counter, vec_b: Counter) -> float:
        """inputs: vec_a: Counter, vec_b: Counter
        說明: 計算兩個詞頻向量的餘弦相似度。
        return: float
        """

        if not vec_a or not vec_b:
            return 0.0

        dot_product = sum(vec_a[key] * vec_b.get(key, 0) for key in vec_a)
        norm_a = sum(value * value for value in vec_a.values()) ** 0.5
        norm_b = sum(value * value for value in vec_b.values()) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def filter(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """inputs: candidates: list[dict[str, Any]]
        說明: 若餘弦相似度高於門檻，保留字數較少的 rewrite。
        return: list[dict[str, Any]]
        """

        # 門檻設定大於 1.0 視為不啟用
        if self.threshold is None or self.threshold > 1.0:
            return candidates

        sorted_candidates = sorted(candidates, key=lambda cand: len(cand.get("rewritten_prompt") or ""))
        kept: List[Dict[str, Any]] = []
        kept_vectors: List[Counter] = []

        for candidate in sorted_candidates:
            text = candidate.get("rewritten_prompt") or ""
            vector = self._tokenize(text)
            is_similar = False
            for existing_vec in kept_vectors:
                if self._cosine(vector, existing_vec) > self.threshold:
                    is_similar = True
                    break
            if not is_similar:
                kept.append(candidate)
                kept_vectors.append(vector)

        return kept


class SFTDataMaker:
    """inputs: results_dir: Path | str, output_path: Path | str, target_filename: str, similarity_threshold: float
    說明: 處理結果檔案並輸出符合條件的 SFT 資料，可選擇性過濾高相似 rewrite。
    return: None
    """

    def __init__(
        self,
        results_dir: Path | str,
        output_path: Path | str,
        target_filename: str,
        similarity_threshold: float,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.output_path = Path(output_path)
        self.target_filename = target_filename
        self.similarity_filter = RewriteSimilarityFilter(similarity_threshold)

    def discover_target_files(self) -> List[Path]:
        """inputs: None
        說明: 搜尋 results 目錄下所有符合檔名的檔案。
        return: list[Path]
        """

        if not self.results_dir.exists():
            raise FileNotFoundError(f"找不到 results 目錄: {self.results_dir}")
        return sorted(self.results_dir.rglob(self.target_filename))

    @staticmethod
    def _is_full_score(value: Any) -> bool:
        """inputs: value: Any
        說明: 判斷分數是否可轉成 1.0。
        return: bool
        """

        try:
            return float(value) == 1.0
        except (TypeError, ValueError):
            return False

    def collect_from_file(self, file_path: Path, bucket: Dict[Any, Dict[str, Any]]) -> None:
        """inputs: file_path: Path, bucket: dict[Any, dict[str, Any]]
        說明: 讀取單一檔案，將滿足條件的 rewrite 以 distill 格式聚合進 bucket。
        return: None
        """

        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    print(f"跳過無法解析的行: {file_path}:{line_no}")
                    continue

                if not self._is_full_score(payload.get("safety_score")):
                    continue
                if not self._is_full_score(payload.get("relevance_score")):
                    continue

                record_id = payload.get("id")
                original_prompt = payload.get("original_prompt")
                rewritten_prompt = payload.get("rewritten_prompt")
                if record_id is None or original_prompt is None or rewritten_prompt is None:
                    continue

                record = bucket.setdefault(
                    record_id,
                    {
                        "id": record_id,
                        "original_prompt": original_prompt,
                        "category": payload.get("category"),
                        "cost": payload.get("cost"),
                        "candidates": [],
                    },
                )

                # 回填缺漏資訊（不同 evaluate 批次可能缺 category/cost）
                if not record.get("original_prompt") and original_prompt:
                    record["original_prompt"] = original_prompt
                if record.get("category") is None and payload.get("category") is not None:
                    record["category"] = payload.get("category")
                if record.get("cost") is None and payload.get("cost") is not None:
                    record["cost"] = payload.get("cost")

                # 避免重複的 rewrite
                if any(candidate.get("rewritten_prompt") == rewritten_prompt for candidate in record["candidates"]):
                    continue

                record["candidates"].append(
                    {
                        "rewritten_prompt": rewritten_prompt,
                        "safety_score": float(payload.get("safety_score", 0.0) or 0.0),
                        "relevance_score": float(payload.get("relevance_score", 0.0) or 0.0),
                        "chat_response": payload.get("chat_response"),
                        "cost": payload.get("cost"),
                    }
                )

    def build_dataset(self, target_files: List[Path]) -> List[Dict[str, Any]]:
        """inputs: target_files: list[Path]
        說明: 彙整多個檔案的 rewrite，輸出與 distill sft.jsonl 相同格式。
        return: list[dict[str, Any]]，每筆包含 id、original_prompt、rewrites
        """

        aggregated: Dict[Any, Dict[str, Any]] = {}
        for path in target_files:
            self.collect_from_file(path, aggregated)

        prepared: List[Dict[str, Any]] = []
        for record in aggregated.values():
            filtered_candidates = self.similarity_filter.filter(record.get("candidates", []))
            rewrites: List[Dict[str, Any]] = []
            for idx, candidate in enumerate(filtered_candidates, start=1):
                rewrites.append(
                    {
                        "index": idx,
                        "rewritten_prompt": candidate.get("rewritten_prompt"),
                    }
                )

            if not rewrites:
                continue

            prepared.append(
                {
                    "id": record.get("id"),
                    "original_prompt": record.get("original_prompt"),
                    "rewrites": rewrites,
                }
            )

        return sorted(prepared, key=lambda rec: rec.get("id"))

    def write_output(self, records: List[Dict[str, Any]]) -> None:
        """inputs: records: list[dict[str, Any]]
        說明: 將 distill 格式的資料寫入指定的 jsonl 輸出檔案。
        return: None
        """

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self) -> None:
        """inputs: None
        說明: 執行完整流程：搜尋檔案、擷取資料並寫入輸出。
        return: None
        """

        target_files = self.discover_target_files()
        if not target_files:
            print(f"未在 {self.results_dir} 下找到任何 {self.target_filename} 檔案。")
            return

        dataset = self.build_dataset(target_files)
        if not dataset:
            print("沒有符合條件的資料可寫入。")
            return

        self.write_output(dataset)
        print(f"已寫入 {len(dataset)} 筆資料到 {self.output_path}")


def parse_args() -> argparse.Namespace:
    """inputs: None
    說明: 解析命令列參數。
    return: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="依評估結果產生 SFT 訓練資料。")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="包含 trial 的 results 根目錄")
    parser.add_argument(
        "--target-name",
        type=str,
        default="raw_adl_final_25w_part1_with_cost.jsonl",
        help="要搜尋的目標檔名",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/sft_v2.jsonl"),
        help="輸出 jsonl 檔案路徑",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=1.01,
        help="rewrite 之間的餘弦相似度門檻，<=1.0 時啟用過濾（預設 1.01 代表關閉）。",
    )
    return parser.parse_args()


def main() -> None:
    """inputs: None
    說明: 建立執行個體並啟動資料處理流程。
    return: None
    """

    args = parse_args()
    maker = SFTDataMaker(args.results_dir, args.output, args.target_name, args.similarity_threshold)
    maker.run()


if __name__ == "__main__":
    main()

"""
python train/sft/make_sft_data.py \
    --results-dir results/part1 \
    --target-name raw_adl_final_25w_part1_with_cost.jsonl \
    --output data/sft/sft_filt.jsonl \
    --similarity-threshold 0.95
"""
