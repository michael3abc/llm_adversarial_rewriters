from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def sanitize_model_name(model_name: str) -> str:
    """
    1. input: 模型名稱（可能含 / 或空白）。
    2. 說明: 將名稱轉成檔名友善格式，只保留字母數字以及 ._-。
    3. output: 字串（若清空則回傳 model）。
    """

    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", model_name.strip())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-") or "model"
    return sanitized


def sanitize_stage_name(stage: str) -> str:
    """
    1. input: stage 字串（stage1/Stage2/...）。
    2. 說明: 轉換成 trial 命名可用的簡潔格式。
    3. output: 小寫+數字的 stage 名稱。
    """

    if not stage:
        return "stage"
    stage = stage.strip().lower()
    stage = re.sub(r"[^0-9a-z]+", "-", stage)
    stage = stage.strip("-") or "stage"
    return stage


@dataclass
class TrialManager:
    """
    1. input: runs 目錄、模型名稱與 stage。
    2. 說明: 尋找下一個 trial_XX_stage_model 目錄。
    3. output: (trial_name, Path)。
    """

    runs_dir: str | Path
    model_name: str
    stage: str = "stage1"

    def next(self) -> Tuple[str, Path]:
        safe_model = sanitize_model_name(self.model_name)
        safe_stage = sanitize_stage_name(self.stage)
        runs_root = Path(self.runs_dir)
        runs_root.mkdir(parents=True, exist_ok=True)

        max_index = 0
        pattern = re.compile(rf"^trial_(\d+)_({re.escape(safe_stage)})_({re.escape(safe_model)})$")
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if not match:
                continue
            max_index = max(max_index, int(match.group(1)))

        next_index = max_index + 1
        trial_name = f"trial_{next_index:02d}_{safe_stage}_{safe_model}"
        return trial_name, runs_root / trial_name


__all__ = ["TrialManager", "sanitize_model_name", "sanitize_stage_name"]
