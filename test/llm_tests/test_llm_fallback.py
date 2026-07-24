import asyncio

import pytest

from tools.llm_fallback import FallbackLLM, should_fallback


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def apredict(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result


class StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_rate_limit_uses_local_model():
    llm = FallbackLLM(
        FakeLLM(error=StatusError(429)),
        lambda: FakeLLM(result="local answer"),
    )

    assert asyncio.run(llm.apredict("question")) == "local answer"


def test_timeout_uses_local_model():
    llm = FallbackLLM(
        FakeLLM(error=TimeoutError("provider timeout")),
        lambda: FakeLLM(result="local answer"),
    )

    assert asyncio.run(llm.apredict("question")) == "local answer"


def test_business_error_is_not_hidden_by_fallback():
    llm = FallbackLLM(
        FakeLLM(error=StatusError(400)),
        lambda: FakeLLM(result="should not be used"),
    )

    with pytest.raises(StatusError):
        asyncio.run(llm.apredict("question"))


def test_only_transient_provider_errors_are_retryable():
    assert should_fallback(ConnectionError("offline"))
    assert should_fallback(StatusError(503))
    assert not should_fallback(ValueError("invalid structured output"))
