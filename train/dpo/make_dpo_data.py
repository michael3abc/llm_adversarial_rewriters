from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


class DPOMaker:
    """inputs: pos_path: Path, neg_path: Path, output_path: Path
    說明: 將正負 rewrites 依 id 聚合並配對，輸出 DPO 訓練資料。
    return: None
    """

    def __init__(self, pos_path: Path, neg_path: Path, output_path: Path) -> None:
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.output_path = output_path

    @staticmethod
    def _load_rewrites(path: Path, kind: str) -> Dict[int, Dict[str, List[str]]]:
        """inputs: path: Path, kind: str
        說明: 讀取 jsonl，依 id 聚合 rewrites 並保留 original_prompt。
        return: dict[id] -> {"prompt": str, "rewrites": list[str]}
        """

        bucket: Dict[int, Dict[str, List[str]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    print(f"跳過無法解析的行: {path}:{line_no}")
                    continue

                record_id = payload.get("id")
                original_prompt = payload.get("original_prompt")
                rewrites = payload.get("rewrites", [])
                if record_id is None or original_prompt is None:
                    print(f"跳過缺少 id/original_prompt 的行: {path}:{line_no}")
                    continue

                entry = bucket.setdefault(record_id, {"prompt": original_prompt, "rewrites": []})
                if not entry.get("prompt"):
                    entry["prompt"] = original_prompt

                for rewrite in rewrites:
                    text = rewrite.get("rewritten_prompt")
                    if text:
                        entry["rewrites"].append(text)

                if not entry["rewrites"]:
                    print(f"Warning: {kind} 無 rewrite 可用，行: {path}:{line_no}")
        return bucket

    def build(self) -> List[Dict[str, object]]:
        """inputs: 無
        說明: 將 pos/neg 依 id 配對，若任一側缺少則略過；使用第一個 pos 搭配所有 neg 產生多筆。
        return: list[dict] 供輸出
        """

        if not self.pos_path.is_file():
            raise FileNotFoundError(f"pos 檔案不存在: {self.pos_path}")
        if not self.neg_path.is_file():
            raise FileNotFoundError(f"neg 檔案不存在: {self.neg_path}")

        pos_bucket = self._load_rewrites(self.pos_path, "pos")
        neg_bucket = self._load_rewrites(self.neg_path, "neg")

        pairs: List[Dict[str, object]] = []
        common_ids = set(pos_bucket.keys()) & set(neg_bucket.keys())
        skipped = (set(pos_bucket.keys()) ^ set(neg_bucket.keys()))
        if skipped:
            print(f"略過 {len(skipped)} 個無法配對的 id。")

        for record_id in sorted(common_ids):
            pos_entry = pos_bucket[record_id]
            neg_entry = neg_bucket[record_id]

            pos_rewrites = pos_entry.get("rewrites") or []
            neg_rewrites = neg_entry.get("rewrites") or []
            if not pos_rewrites or not neg_rewrites:
                continue

            chosen = pos_rewrites[0]
            original_prompt: Optional[str] = pos_entry.get("prompt") or neg_entry.get("prompt")

            for rejected in neg_rewrites:
                pairs.append(
                    {
                        "id": record_id,
                        "original_prompt": original_prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

        print(f"產生 DPO 筆數: {len(pairs)}（配對 id 數: {len(common_ids)}）")
        return pairs

    def write(self, records: List[Dict[str, object]]) -> None:
        """inputs: records: list[dict]
        說明: 將 DPO 記錄寫入 jsonl。
        return: None
        """

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"已寫入 {len(records)} 筆至 {self.output_path}")


def parse_args() -> argparse.Namespace:
    """inputs: None
    說明: 解析命令列參數。
    return: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="依 pos/neg rewrites 產生 DPO 訓練資料。")
    parser.add_argument("--pos", type=Path, default=Path("data/sft/pos_sample.jsonl"), help="正樣本路徑。")
    parser.add_argument("--neg", type=Path, default=Path("data/sft/neg_sample.jsonl"), help="負樣本路徑。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dpo/dpo_train_data.jsonl"),
        help="輸出 jsonl 路徑。",
    )
    return parser.parse_args()


def main() -> None:
    """inputs: None
    說明: 執行 DPO 配對流程。
    return: None
    """

    args = parse_args()
    maker = DPOMaker(args.pos, args.neg, args.output)
    records = maker.build()
    maker.write(records)


if __name__ == "__main__":
    main()

"""
python train/dpo/make_dpo_data.py \
    --pos data/sft/pos_sample.jsonl \
    --neg data/sft/neg_sample.jsonl \
    --output data/dpo/dpo_train_data.jsonl
"""