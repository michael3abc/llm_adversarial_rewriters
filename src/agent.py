from typing import Callable, Optional

from . import algorithms  # Import the participant's algorithms file


class PromptSafetyAgent:
    """Agent wrapper around algorithms.py functions with optional model adapter overrides."""

    # The required function name for the final submission
    MANDATORY_ENTRY_POINT = "evaluate_rewrite"

    def __init__(
        self,
        algorithm_name: str,
        *,
        adapter_path: Optional[str] = None,
        base_model: Optional[str] = None,
        adapter_device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        1. inputs: algorithm_name 以及可選的 adapter_path、base_model、adapter_device、max_new_tokens、temperature、top_p、system_prompt。
        2. 說明: 動態載入 algorithms.py 中的函式，並在提供參數時同步覆寫 rewrite adapter 設定。
        3. return: None。
        """

        self.algorithm_name = algorithm_name
        self._rewrite_function: Optional[Callable[[str], str]] = None
        self._configure_adapter(
            adapter_path=adapter_path,
            base_model=base_model,
            device=adapter_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        )

        try:
            self._rewrite_function = getattr(algorithms, algorithm_name)
            print(f"PromptSafetyAgent initialized with algorithm: {algorithm_name}")
        except AttributeError as exc:
            raise ValueError(
                f"Algorithm '{algorithm_name}' not found in algorithms.py. "
                f"Make sure the function name is correct."
            ) from exc

    def _configure_adapter(self, **overrides) -> None:
        """
        1. inputs: overrides - adapter 設定覆寫（僅保留值非 None 的欄位）。
        2. 說明: 若 algorithms 模組提供 configure_model_adapter，就將 overrides 套用進去。
        3. return: None。
        """

        if not hasattr(algorithms, "configure_model_adapter"):
            return
        filtered = {key: value for key, value in overrides.items() if value is not None}
        if filtered:
            algorithms.configure_model_adapter(**filtered)

    def rewrite(self, toxic_prompt: str, category: Optional[str] = None) -> str:
        """
        1. inputs: toxic_prompt: str；category: Optional[str]。
        2. 說明: 執行載入的演算法函式；若未載入成功則直接回傳原 prompt。若提供 category，會嘗試傳入演算法（few-shot 用途）。
        3. return: str。
        """

        if not self._rewrite_function:
            return toxic_prompt
        if category is None:
            return self._rewrite_function(toxic_prompt)
        try:
            return self._rewrite_function(toxic_prompt, category=category)
        except TypeError:
            return self._rewrite_function(toxic_prompt)
