from __future__ import annotations

import argparse
import json
import math
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class RewriteRecord:
    category: str
    safety_score: float
    relevance_score: float


@dataclass
class CategoryStat:
    num: int
    mean_safety_score: float
    safety_std: float
    safety_ratio: float
    mean_relevance_score: float
    relevance_std: float

    def to_serializable(self) -> Dict[str, float | int]:
        """
        1. inputs: 無
        2. 說明: 將統計數值轉成可寫入 JSON 的字典。
        3. return: dict，包含所有欄位。
        """
        payload = asdict(self)
        payload["safty_std"] = self.safety_std
        payload["safty_ratio"] = self.safety_ratio
        return payload


class StatisticsAnalyzer:
    def __init__(self, original_path: Path, rewritten_path: Path, output_dir: Path):
        """
        1. inputs: original_path (原始 JSONL 路徑), rewritten_path (rewrite JSONL 路徑), output_dir (輸出資料夾)
        2. 說明: 初始化檔案路徑並建立輸出資料夾。
        3. return: None
        """
        self.original_path = original_path
        self.rewritten_path = rewritten_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._category_order: List[str] = []

    def _read_jsonl(self, path: Path) -> List[dict]:
        """
        1. inputs: path (JSONL 檔案路徑)
        2. 說明: 逐行讀取 JSONL 並轉成 list[dict]。
        3. return: 解析後的 list。
        """
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _load_originals(self) -> Dict[int, dict]:
        """
        1. inputs: 無
        2. 說明: 載入原始資料，並建立 question_id 到紀錄的對照表。
        3. return: dict，key 為 question_id，value 為原始紀錄。
        """
        originals = self._read_jsonl(self.original_path)
        self._category_order = self._collect_category_order(originals)
        return {item["question_id"]: item for item in originals}

    def _collect_category_order(self, originals: List[dict]) -> List[str]:
        """
        1. inputs: originals (原始紀錄列表)
        2. 說明: 依檔案出現順序紀錄各 category，方便統一繪圖順序。
        3. return: category 名稱的清單。
        """
        seen = set()
        ordered = []
        for item in originals:
            category = item.get("category", "unknown")
            if category in seen:
                continue
            seen.add(category)
            ordered.append(category)
        return ordered

    def _load_rewrites(self) -> List[dict]:
        """
        1. inputs: 無
        2. 說明: 載入 rewrite JSONL。
        3. return: rewrite 紀錄的 list。
        """
        return self._read_jsonl(self.rewritten_path)

    def _align_records(self, originals: Dict[int, dict], rewrites: List[dict]) -> List[RewriteRecord]:
        """
        1. inputs: originals (question_id 對應的原始資料), rewrites (rewrite 紀錄列表)
        2. 說明: 將 rewrite 與原始資料以 question_id-1 對齊，產出含 category 的紀錄。
        3. return: RewriteRecord 物件列表。
        """
        aligned: List[RewriteRecord] = []
        for rewrite in rewrites:
            question_id = rewrite["id"] + 1
            original = originals.get(question_id)
            if not original:
                continue
            aligned.append(
                RewriteRecord(
                    category=original.get("category", "unknown"),
                    safety_score=float(rewrite.get("safety_score", 0.0)),
                    relevance_score=float(rewrite.get("relevance_score", 0.0)),
                )
            )
        return aligned

    @staticmethod
    def _mean(values: List[float]) -> float:
        """
        1. inputs: values (浮點數列表)
        2. 說明: 計算平均值，空列表時回傳 0。
        3. return: 平均值。
        """
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _std(values: List[float]) -> float:
        """
        1. inputs: values (浮點數列表)
        2. 說明: 計算母體標準差，長度小於 2 則回傳 0。
        3. return: 標準差。
        """
        if len(values) < 2:
            return 0.0
        mean_value = StatisticsAnalyzer._mean(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(math.sqrt(variance))

    def _normalize_counts(
        self, counts: Dict[float, List[int]], score_levels: List[float]
    ) -> Dict[float, List[float]]:
        """
        1. inputs: counts (分數到各 category 計數的對照表), score_levels (分數層級列表)
        2. 說明: 依各 category 總數將計數轉為比例，避免樣本數差異影響視覺比較。
        3. return: dict，key 為分數，value 為各 category 的比例列表。
        """
        normalized: Dict[float, List[float]] = {score: [] for score in score_levels}
        category_count = len(next(iter(counts.values()), []))

        for idx in range(category_count):
            total = sum(counts[score][idx] for score in score_levels)
            for score in score_levels:
                normalized[score].append(counts[score][idx] / total if total else 0.0)
        return normalized

    def _compute_category_stats(self, records: List[RewriteRecord]) -> Dict[str, CategoryStat]:
        """
        1. inputs: records (對齊後的 RewriteRecord 列表)
        2. 說明: 依 category 彙整數量、平均值、標準差與安全比例。
        3. return: dict，key 為 category，value 為 CategoryStat。
        """
        safety_bucket: Dict[str, List[float]] = {}
        relevance_bucket: Dict[str, List[float]] = {}

        for record in records:
            safety_bucket.setdefault(record.category, []).append(record.safety_score)
            relevance_bucket.setdefault(record.category, []).append(record.relevance_score)

        stats: Dict[str, CategoryStat] = {}
        all_categories = self._category_order or sorted(safety_bucket.keys())

        for category in all_categories:
            safety_scores = safety_bucket.get(category, [])
            relevance_scores = relevance_bucket.get(category, [])
            count = len(safety_scores)
            safety_ratio = (
                float(sum(1 for score in safety_scores if score == 1.0) / count) if count else 0.0
            )
            stats[category] = CategoryStat(
                num=count,
                mean_safety_score=self._mean(safety_scores),
                safety_std=self._std(safety_scores),
                safety_ratio=safety_ratio,
                mean_relevance_score=self._mean(relevance_scores),
                relevance_std=self._std(relevance_scores),
            )
        return stats

    def _plot_safety(self, records: List[RewriteRecord]) -> Path:
        """
        1. inputs: records (RewriteRecord 列表)
        2. 說明: 以長條圖展示各 category 在 safety_score 0、0.5、1 的比例分佈。
        3. return: 產出的圖檔路徑。
        """
        categories = self._category_order or sorted({r.category for r in records})
        score_levels = [0.0, 0.5, 1.0]
        counts = {score: [0 for _ in categories] for score in score_levels}

        for idx, category in enumerate(categories):
            category_scores = [r.safety_score for r in records if r.category == category]
            for score in score_levels:
                counts[score][idx] = sum(1 for value in category_scores if value == score)

        normalized_counts = self._normalize_counts(counts, score_levels)
        x_positions = list(range(len(categories)))
        bar_width = 0.25

        plt.figure(figsize=(16, 9))
        for offset, score in enumerate(score_levels):
            plt.bar(
                [x + offset * bar_width for x in x_positions],
                normalized_counts[score],
                width=bar_width,
                label=f"score {score}",
            )

        plt.xlabel("Category")
        plt.ylabel("Ratio")
        plt.title("Normalized Safety Score Distribution by Category")
        plt.xticks(
            [x + bar_width for x in x_positions],
            categories,
            rotation=30,
            ha="right",
            fontsize=7,
        )
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / "safety_bar_plot.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    def _plot_relevance(self, records: List[RewriteRecord]) -> Path:
        """
        1. inputs: records (RewriteRecord 列表)
        2. 說明: 以長條圖展示各 category 在 relevance_score 0、1 的比例分佈。
        3. return: 產出的圖檔路徑。
        """
        categories = self._category_order or sorted({r.category for r in records})
        score_levels = [0.0, 1.0]
        counts = {score: [0 for _ in categories] for score in score_levels}

        for idx, category in enumerate(categories):
            category_scores = [r.relevance_score for r in records if r.category == category]
            for score in score_levels:
                counts[score][idx] = sum(1 for value in category_scores if value == score)

        normalized_counts = self._normalize_counts(counts, score_levels)
        x_positions = list(range(len(categories)))
        bar_width = 0.35

        plt.figure(figsize=(16, 9))
        for offset, score in enumerate(score_levels):
            plt.bar(
                [x + offset * bar_width for x in x_positions],
                normalized_counts[score],
                width=bar_width,
                label=f"score {score}",
            )

        plt.xlabel("Category")
        plt.ylabel("Ratio")
        plt.title("Normalized Relevance Score Distribution by Category")
        plt.xticks(
            [x + bar_width * 0.5 for x in x_positions],
            categories,
            rotation=30,
            ha="right",
            fontsize=7,
        )
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / "relevance_bar_plot.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    def _save_stats(self, stats: Dict[str, CategoryStat]) -> Path:
        """
        1. inputs: stats (各 category 的統計結果)
        2. 說明: 將統計數值寫入 JSON 方便後續分析。
        3. return: 輸出檔案路徑。
        """
        output_path = self.output_dir / "category_statistics.json"
        serializable = {category: stat.to_serializable() for category, stat in stats.items()}
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
        return output_path

    def _save_markdown(self, stats: Dict[str, CategoryStat]) -> Path:
        """
        1. inputs: stats (各 category 的統計結果)
        2. 說明: 將統計數值輸出為 Markdown 表格，方便閱讀。
        3. return: 輸出檔案路徑。
        """
        output_path = self.output_dir / "category_statistics.md"
        header = [
            "category",
            "num",
            "mean_safety_score",
            "safety_std",
            "safety_ratio",
            "mean_relevance_score",
            "relevance_std",
        ]
        lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]

        # 依照原始順序輸出，若不存在則使用排序後的鍵
        ordered_categories = self._category_order or sorted(stats.keys())
        for category in ordered_categories:
            stat = stats.get(category)
            if not stat:
                continue
            values = [
                category,
                str(stat.num),
                f"{stat.mean_safety_score:.4f}",
                f"{stat.safety_std:.4f}",
                f"{stat.safety_ratio:.4f}",
                f"{stat.mean_relevance_score:.4f}",
                f"{stat.relevance_std:.4f}",
            ]
            lines.append("|" + "|".join(values) + "|")

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return output_path

    def _save_csv(self, stats: Dict[str, CategoryStat]) -> Path:
        """
        1. inputs: stats (各 category 的統計結果)
        2. 說明: 將統計數值輸出為 CSV，便於試算或後續分析。
        3. return: 輸出檔案路徑。
        """
        output_path = self.output_dir / "category_statistics.csv"
        header = [
            "category",
            "num",
            "mean_safety_score",
            "safety_std",
            "safety_ratio",
            "mean_relevance_score",
            "relevance_std",
        ]
        ordered_categories = self._category_order or sorted(stats.keys())

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for category in ordered_categories:
                stat = stats.get(category)
                if not stat:
                    continue
                writer.writerow(
                    [
                        category,
                        stat.num,
                        f"{stat.mean_safety_score:.4f}",
                        f"{stat.safety_std:.4f}",
                        f"{stat.safety_ratio:.4f}",
                        f"{stat.mean_relevance_score:.4f}",
                        f"{stat.relevance_std:.4f}",
                    ]
                )
        return output_path

    def run(self) -> Tuple[Dict[str, CategoryStat], Path, Path, Path, Path, Path]:
        """
        1. inputs: 無
        2. 說明: 執行完整流程：讀檔、對齊、統計、繪圖與輸出。
        3. return: (統計結果 dict, 統計 JSON 路徑, Markdown 路徑, CSV 路徑, safety 圖檔路徑, relevance 圖檔路徑)
        """
        originals = self._load_originals()
        rewrites = self._load_rewrites()
        records = self._align_records(originals, rewrites)
        stats = self._compute_category_stats(records)
        stats_path = self._save_stats(stats)
        md_path = self._save_markdown(stats)
        csv_path = self._save_csv(stats)
        safety_plot = self._plot_safety(records)
        relevance_plot = self._plot_relevance(records)
        return stats, stats_path, md_path, csv_path, safety_plot, relevance_plot


def parse_args() -> argparse.Namespace:
    """
    1. inputs: 無
    2. 說明: 解析 CLI 參數以指定輸入路徑，輸出固定寫在 rewrite 同資料夾。
    3. return: argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="Calculate statistics and plots for rewrites.")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/adl_final_25w_part1_with_cost.jsonl"),
        help="原始 JSONL 檔案路徑。",
    )
    parser.add_argument(
        "--rewrite",
        type=Path,
        default=Path("results/evaluate_rewrite_01/raw_adl_final_25w_part1_with_cost.jsonl"),
        help="rewrite JSONL 檔案路徑。",
    )
    return parser.parse_args()


def main():
    """
    1. inputs: 無
    2. 說明: 透過 CLI 執行統計流程並印出輸出路徑。
    3. return: None
    """
    args = parse_args()
    output_dir = args.rewrite.parent
    analyzer = StatisticsAnalyzer(args.original, args.rewrite, output_dir)
    stats, stats_path, md_path, csv_path, safety_plot, relevance_plot = analyzer.run()

    print(f"統計結果已寫入: {stats_path}")
    print(f"Markdown 輸出: {md_path}")
    print(f"CSV 輸出: {csv_path}")
    print(f"Safety 圖檔: {safety_plot}")
    print(f"Relevance 圖檔: {relevance_plot}")
    print("各 category 數據：")
    for category, stat in stats.items():
        print(f"- {category}: {stat.to_serializable()}")


if __name__ == "__main__":
    main()


"""
python src/utils/statistics.py \
    --original data/adl_final_25w_part1_with_cost.jsonl \
    --rewrite results/evaluate_rewrite_04/raw_adl_final_25w_part1_with_cost.jsonl
"""
