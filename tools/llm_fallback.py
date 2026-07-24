from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {
    "apiconnectionerror",
    "apitimeouterror",
    "ratelimiterror",
    "serviceunavailableerror",
}


def _status_code(error: BaseException) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def should_fallback(error: BaseException) -> bool:
    """Return whether a provider failure is safe to retry on the local model."""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if _status_code(error) in _TRANSIENT_STATUS_CODES:
        return True
    return type(error).__name__.lower() in _TRANSIENT_NAMES


class FallbackLLM:
    """OpenAI/LlamaIndex-compatible async facade with a lazy local fallback."""

    def __init__(
        self,
        primary: Any,
        fallback_factory: Callable[[], Any],
    ) -> None:
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: Any | None = None

    @property
    def fallback(self) -> Any:
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return await getattr(self.primary, method)(*args, **kwargs)
        except Exception as error:
            if not should_fallback(error):
                raise
            logger.warning(
                "Cloud LLM failed with %s; falling back to local inference",
                type(error).__name__,
            )
            return await getattr(self.fallback, method)(*args, **kwargs)

    async def apredict(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("apredict", *args, **kwargs)

    async def astructured_predict(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("astructured_predict", *args, **kwargs)

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("achat", *args, **kwargs)

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("acomplete", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)
