"""LLM wrapper: the library's GeneralLlm plus spend metering. Import only after envguard.apply()."""

from __future__ import annotations

import inspect
import logging
from typing import Callable

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from bot.config import Config, is_search_model, normalise_model_name
from bot.envguard import redact
from bot.spend import BudgetExceeded, SpendMeter

logger = logging.getLogger(__name__)

# Charged when a call fails after being sent (timeouts are often billed by the provider).
_FAILED_CALL_TOKENS = (2000, 2000)

# (model name, prompt) -> answer text (or an awaitable of it), or raises. Used by tests and by the offline --fake-llm mode.
Responder = Callable[[str, str], str]


class EmptyAnswer(RuntimeError):
    """The provider answered, but with no answer text (e.g. the token ceiling was spent on reasoning).

    Carries the call's real usage so the meter charges what was actually billed.
    """

    def __init__(self, message: str, prompt_tokens: int, completion_tokens: int, cost: float | None) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost


def _finish_details(response) -> tuple[str, int, int, float | None]:
    """(one-line description, prompt tokens, completion tokens, reported cost) of a litellm response."""
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    native = getattr(choice, "native_finish_reason", None) or (
        getattr(choice, "provider_specific_fields", None) or {}
    ).get("native_finish_reason")
    provider = (getattr(response, "model_extra", None) or {}).get("provider")
    cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
    text = (
        f"finish_reason={choice.finish_reason} native_finish_reason={native} "
        f"completion_tokens={completion_tokens} reasoning_tokens={reasoning_tokens} provider={provider}"
    )
    return text, prompt_tokens, completion_tokens, cost


class TrackedLlm(GeneralLlm):
    """GeneralLlm whose every provider call is checked against, and recorded in, a SpendMeter."""

    def __init__(self, model: str, meter: SpendMeter | None = None, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._meter = meter

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        # Checked here, outside the library's retry loop, so an exhausted budget is not "retried".
        if self._meter is not None:
            self._meter.check()
        return await super().invoke(prompt, system_prompt)

    async def _call_litellm(self, prompt):  # type: ignore[override]
        response = await super()._call_litellm(prompt)
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            if response.choices[0].finish_reason == "length":
                logger.warning("%s answer was cut off at the token ceiling (%s)", self.model, _finish_details(response)[0])
            return response
        # The library would fall back to the reasoning text here. For these models that text is a
        # summary of the model's thinking, not an answer: a number parsed from it is a guess. Fail
        # instead, so the call is retried and then handed to another model.
        description, prompt_tokens, completion_tokens, cost = _finish_details(response)
        logger.warning("%s returned no answer text (%s)", self.model, description)
        raise EmptyAnswer(
            f"{self.model} returned no answer text ({description})", prompt_tokens, completion_tokens, cost
        )

    async def _raw_call(self, prompt) -> TextTokenCostResponse:
        return await super()._mockable_direct_call_to_model(prompt)

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        meter = self._meter
        if meter is not None:
            meter.before_call(self.model)
        try:
            response = await self._raw_call(prompt)
        except BudgetExceeded:
            raise
        except EmptyAnswer as error:
            if meter is not None:
                meter.after_call(self.model, error.prompt_tokens, error.completion_tokens, error.cost)
            raise
        except Exception as error:
            # One line per failed attempt: the library retries these without logging them.
            logger.warning("%s call failed: %s: %s", self.model, error.__class__.__name__, redact(str(error))[:300])
            if meter is not None:
                meter.after_call(self.model, *_FAILED_CALL_TOKENS, None)
            raise
        if meter is not None:
            meter.after_call(
                self.model,
                response.prompt_tokens_used,
                response.completion_tokens_used,
                response.cost,
            )
        return response


class ResponderLlm(TrackedLlm):
    """Never touches the network: answers come from a local function. Metering still applies."""

    def __init__(self, model: str, responder: Responder, meter: SpendMeter | None = None, **kwargs) -> None:
        super().__init__(model, meter=meter, **kwargs)
        self._responder = responder

    async def _raw_call(self, prompt) -> TextTokenCostResponse:
        text = self._responder(self.model, prompt if isinstance(prompt, str) else str(prompt))
        if inspect.isawaitable(text):
            text = await text
        if not isinstance(text, str) or text == "":
            raise RuntimeError("LLM answer is an empty string")
        prompt_tokens = max(1, len(str(prompt)) // 4)
        completion_tokens = max(1, len(text) // 4)
        return TextTokenCostResponse(
            data=text,
            prompt_tokens_used=prompt_tokens,
            completion_tokens_used=completion_tokens,
            total_tokens_used=prompt_tokens + completion_tokens,
            model=self.model,
            cost=0.0,
        )


class LlmFactory:
    """Builds (and caches) TrackedLlm objects from config. One instance per run."""

    def __init__(self, config: Config, meter: SpendMeter | None, responder: Responder | None = None) -> None:
        self.config = config
        self.meter = meter
        self.responder = responder
        self._cache: dict[tuple[str, str], TrackedLlm] = {}

    @property
    def is_fake(self) -> bool:
        return self.responder is not None

    def _build(self, model: str, role: str) -> TrackedLlm:
        models = self.config.models
        timeout = {
            "forecast": models.forecast_timeout_s,
            "search": models.search_timeout_s,
            "parser": models.parser_timeout_s,
        }[role]
        kwargs: dict = {
            "temperature": None,  # current reasoning models reject it; litellm drops None
            "timeout": timeout,
            "allowed_tries": models.allowed_tries_by_model.get(normalise_model_name(model), models.allowed_tries),
            "max_tokens": models.max_output_tokens,
        }
        if role == "forecast" and models.reasoning_effort:
            kwargs["reasoning_effort"] = models.reasoning_effort
        if role == "parser":
            kwargs["max_tokens"] = 2000
        if self.responder is not None:
            return ResponderLlm(model, self.responder, meter=self.meter, **kwargs)
        return TrackedLlm(model, meter=self.meter, **kwargs)

    def get(self, model: str, role: str) -> TrackedLlm:
        key = (model, role)
        if key not in self._cache:
            self._cache[key] = self._build(model, role)
        return self._cache[key]

    def forecaster(self, model: str) -> TrackedLlm:
        return self.get(model, "forecast")

    def searcher(self, model: str) -> TrackedLlm:
        if not is_search_model(model):
            logger.warning("search model %s has no ':online' suffix; it will answer from memory only", model)
        return self.get(model, "search")

    def parser(self) -> TrackedLlm:
        return self.get(self.config.models.parser, "parser")
