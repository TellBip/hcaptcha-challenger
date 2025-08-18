import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Generic

from google.genai.types import ThinkingConfig
from loguru import logger

from hcaptcha_challenger.models import THINKING_BUDGET_MODELS
from hcaptcha_challenger.tools.common import run_sync

M = TypeVar("M")


class _Reasoner(ABC, Generic[M]):

    def __init__(
        self, gemini_api_key: str, model: M | None = None, constraint_response_schema: bool = False
    ):
        self._api_key: str = gemini_api_key
        self._model: M | None = model
        self._constraint_response_schema = constraint_response_schema
        self._response = None

    def cache_response(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(e)

    @abstractmethod
    async def invoke_async(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _pin_thinking_config(
        model_to_use: str, thinking_budget: int | None = None
    ) -> ThinkingConfig | None:
        try:
            if model_to_use not in THINKING_BUDGET_MODELS or not isinstance(thinking_budget, int):
                return
            # Must turn on spatial reasoning
            if model_to_use.startswith("gemini-2.5-pro") and thinking_budget == 0:
                thinking_budget = -1
            if thinking_budget < -1 or thinking_budget > 32768:
                thinking_budget = -1
            return ThinkingConfig(include_thoughts=False, thinking_budget=thinking_budget)
        except Exception as err:
            logger.warning(f"Error resetting thinking config: {str(err)}")

    # for backward compatibility
    def invoke(self, *args, **kwargs):
        return run_sync(self.invoke_async(*args, **kwargs))
