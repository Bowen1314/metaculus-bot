"""TrackedLlm on a provider response with no answer text (seen from Gemini in the first real dry run)."""

from __future__ import annotations

import asyncio

import pytest
from forecasting_tools import GeneralLlm
from litellm import ModelResponse

from bot.llm import EmptyAnswer, LlmFactory
from bot.spend import SpendMeter

GEMINI = "openrouter/google/gemini-3.8-flash"


def response(content, finish_reason="stop", completion_tokens=500) -> ModelResponse:
    return ModelResponse(
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": content, "reasoning_content": "**Calculating** Probability: 40%"},
            "finish_reason": finish_reason,
            "native_finish_reason": "MAX_TOKENS" if finish_reason == "length" else "STOP",
        }],
        usage={"prompt_tokens": 9000, "completion_tokens": completion_tokens, "total_tokens": 9000 + completion_tokens,
               "completion_tokens_details": {"reasoning_tokens": completion_tokens - 10}},
        model="google/gemini-3.8-flash",
        provider="Google AI Studio",
    )


@pytest.fixture()
def gemini(config, ledger, monkeypatch):
    """A real TrackedLlm whose provider call returns whatever `replies` holds next (no network)."""
    replies: list[ModelResponse] = []

    async def fake_call(self, prompt):
        return replies.pop(0)

    monkeypatch.setattr(GeneralLlm, "_call_litellm", fake_call)
    meter = SpendMeter(config.limits, config.prices, ledger)
    return LlmFactory(config, meter).forecaster(GEMINI), replies, meter


def test_reasoning_text_is_not_taken_as_the_answer(gemini, caplog):
    llm, replies, meter = gemini
    replies.append(response(None, finish_reason="length", completion_tokens=20000))
    with pytest.raises(EmptyAnswer, match="finish_reason=length .*reasoning_tokens=19990"):
        asyncio.run(llm.invoke("forecast this"))
    assert "returned no answer text" in caplog.text
    # charged for the 20,000 tokens actually used, not the small flat allowance for failed calls
    assert meter.run_usd == pytest.approx((9000 * 0.75 + 20000 * 3.75) / 1e6)


def test_blank_answer_is_retried_then_succeeds(gemini, fast_sleep):
    llm, replies, meter = gemini
    llm.allowed_tries = 2
    replies += [response("   "), response("Probability: 35%")]
    assert asyncio.run(llm.invoke("forecast this")) == "Probability: 35%"
    assert meter.run_calls == 2 and not replies


def test_normal_answer_passes_through(gemini):
    llm, replies, _ = gemini
    replies.append(response("Percentile 10: 1\nPercentile 90: 9"))
    assert asyncio.run(llm.invoke("forecast this")) == "Percentile 10: 1\nPercentile 90: 9"


def test_per_model_tries_override(config, ledger):
    import dataclasses

    cfg = dataclasses.replace(config, models=dataclasses.replace(
        config.models, allowed_tries=2, allowed_tries_by_model={"google/gemini-3.8-flash": 3}))
    factory = LlmFactory(cfg, SpendMeter(cfg.limits, cfg.prices, ledger))
    assert factory.forecaster(GEMINI).allowed_tries == 3
    assert factory.forecaster("openrouter/openai/gpt-6.1-sol").allowed_tries == 2
