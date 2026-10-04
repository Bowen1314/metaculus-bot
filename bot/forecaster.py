"""
The forecasting bot: the official FallTemplateBot2026 with a cross-lab ensemble.

Kept from the template: the forecasting prompts, one research report shared by all runs, the
LLM-based answer parser (now only a fallback), the comment format.
Changed: several research sources, each run uses a named model with fail-over to another lab,
deterministic parsing first, percentile clean-up, aggregation (bot/aggregate.py, bot/numeric.py).

This class never submits anything: `publish_reports_to_metaculus` is forced to False and the
runner does the submitting, behind the ledger.
"""

from __future__ import annotations

import contextvars
import logging
from datetime import datetime, timezone
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.binary_report import BinaryPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import DatePercentile, Percentile
from forecasting_tools.data_models.questions import ConditionalQuestion, DateQuestion
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import FallTemplateBot2026
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

from bot import aggregate, numeric, parsing
from bot.config import Config, Profile
from bot.envguard import redact
from bot.guard import time_limit
from bot.llm import LlmFactory
from bot.research import Researcher
from bot.spend import BudgetExceeded

logger = logging.getLogger(__name__)

# Models to try, in order, for the forecasting run executing in the current task.
_run_models: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("run_models", default=())

# Per-run clamp before aggregation (the template's own values).
_RUN_MIN, _RUN_MAX = 0.01, 0.99
CDF_TIME_LIMIT_S = 10.0


class UnsupportedQuestion(ValueError):
    pass


def question_key(question: MetaculusQuestion) -> int:
    if question.id_of_question is None:
        raise UnsupportedQuestion("question has no question id")
    return int(question.id_of_question)


class EnsembleBot(FallTemplateBot2026):
    _structure_output_validation_samples = 2

    def __init__(
        self,
        *,
        config: Config,
        profile: Profile,
        factory: LlmFactory,
        researcher: Researcher,
        metaculus_client: Any,
    ) -> None:
        self.cfg = config
        self.profile = profile
        self.factory = factory
        self.researcher = researcher
        self._slots: dict[int, int] = {}
        self.run_details: dict[int, list[dict[str, Any]]] = {}
        self.final_cdfs: dict[int, list[float]] = {}
        parser = factory.parser()
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=len(profile.forecasters),
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=False,
            enable_summarize_research=False,
            extra_metadata_in_explanation=False,
            required_successful_predictions=config.ensemble.min_successful_fraction,
            llms={
                "default": factory.forecaster(profile.forecasters[0]),
                "summarizer": parser,
                "researcher": "no_research",  # research is done by bot/research.py
                "parser": parser,
            },
            metaculus_client=metaculus_client,
        )

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        # The template's defaults fall back to gpt-4o chosen from whatever keys are in the
        # environment. Nothing here may be picked implicitly, so there are no defaults.
        return {"default": None, "summarizer": None, "researcher": None, "parser": None}

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "publish_reports_to_metaculus" and value:
            raise AttributeError("EnsembleBot never publishes; the runner submits through the ledger")
        super().__setattr__(name, value)

    # ------------------------------------------------------------------ research

    async def run_research(self, question: MetaculusQuestion) -> str:
        return await self.researcher.run(question, self.profile)

    # ------------------------------------------------------------------ model choice

    def _models_for_slot(self, slot: int) -> tuple[str, ...]:
        forecasters = self.profile.forecasters
        primary = forecasters[slot % len(forecasters)]
        others: list[str] = []
        for offset in range(1, len(forecasters)):
            candidate = forecasters[(slot + offset) % len(forecasters)]
            if candidate != primary and candidate not in others:
                others.append(candidate)
        return (primary, *others[: max(0, self.cfg.ensemble.model_failover)])

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        if isinstance(question, ConditionalQuestion):
            raise UnsupportedQuestion("conditional questions are not supported")
        key = question_key(question)
        slot = self._slots.get(key, 0)  # no await between read and write: slots are unique
        self._slots[key] = slot + 1
        token = _run_models.set(self._models_for_slot(slot))
        try:
            return await super()._make_prediction(question, research)
        finally:
            _run_models.reset(token)

    async def _invoke_forecaster(self, prompt: str) -> tuple[str, str]:
        models = _run_models.get() or (self.profile.forecasters[0],)
        last_error: Exception | None = None
        for model in models:
            try:
                text = await self.factory.forecaster(model).invoke(prompt)
                if text and text.strip():
                    return text, model
                last_error = RuntimeError("empty answer")
            except BudgetExceeded:
                raise
            except Exception as error:  # timeout, provider error, rate limit, ...
                last_error = error
            logger.warning(
                "forecast call failed on %s: %s: %s",
                model,
                last_error.__class__.__name__,
                redact(str(last_error))[:300],
            )
        assert last_error is not None
        raise last_error

    def _note_run(self, question: MetaculusQuestion, model: str, readable: str, parsed_by: str) -> None:
        self.run_details.setdefault(question_key(question), []).append(
            {"model": model.removeprefix("openrouter/"), "prediction": readable, "parsed_by": parsed_by}
        )

    # ------------------------------------------------------------------ binary

    async def _binary_prompt_to_forecast(self, question: BinaryQuestion, prompt: str) -> ReasonedPrediction[float]:
        reasoning, model = await self._invoke_forecaster(prompt)
        probability = parsing.parse_binary_probability(reasoning)
        parsed_by = "regex"
        if probability is None:
            parsed_by = "llm"
            instructions = clean_indents(
                f"""
                The text given to you is trying to give a probability forecast for a binary question.
                {self._create_resolved_question_parsing_message()}
                """
            )
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=instructions,
            )
            probability = parsed.prediction_in_decimal
        probability = aggregate.clip_probability(probability, low=_RUN_MIN, high=_RUN_MAX)
        self._note_run(question, model, f"{probability:.1%}", parsed_by)
        return ReasonedPrediction(prediction_value=probability, reasoning=reasoning)

    # ------------------------------------------------------------------ multiple choice

    async def _multiple_choice_prompt_to_forecast(
        self, question: MultipleChoiceQuestion, prompt: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        reasoning, model = await self._invoke_forecaster(prompt)
        options = list(question.options)
        probabilities = parsing.parse_option_probabilities(reasoning, options)
        parsed_by = "regex"
        if probabilities is None:
            parsed_by = "llm"
            instructions = clean_indents(
                f"""
                Make sure that all option names are one of the following:
                {question.options}

                The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
                Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
                {self._create_resolved_question_parsing_message()}
                """
            )
            parsed: PredictedOptionList = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=instructions,
            )
            probabilities = {o.option_name: o.probability for o in parsed.predicted_options}
            if set(probabilities) != set(options):
                raise ValueError(f"parsed options {sorted(probabilities)} do not match the question's options")
        probabilities = aggregate.floor_and_normalise(probabilities, floor=_RUN_MIN)
        prediction = _option_list(options, probabilities)
        readable = ", ".join(f"{o}: {probabilities[o]:.0%}" for o in options)
        self._note_run(question, model, readable, parsed_by)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ------------------------------------------------------------------ numeric, discrete

    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning, model = await self._invoke_forecaster(prompt)
        lower, upper = float(question.lower_bound), float(question.upper_bound)
        pairs = parsing.parse_percentiles(reasoning)
        parsed_by = "regex"
        if pairs is not None and not _plausible(pairs, lower, upper):
            # Probably stated in other units (e.g. dollars when the question is in billions);
            # the LLM parser is told the units and converts.
            pairs = None
        if pairs is None:
            parsed_by = "llm"
            instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a numeric question.
                - This text is trying to answer the numeric question: "{question.question_text}".
                {self._create_single_distribution_parsing_message(question)}
                {self._create_resolved_question_parsing_message()}
                - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
                - The units for the forecast are: {question.unit_of_measure}
                - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
                - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
                - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                - Turn any values that are in scientific notation into regular numbers.
                """
            )
            parsed: list[Percentile] = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=instructions,
                num_validation_samples=self._structure_output_validation_samples,
            )
            pairs = [(p.percentile, p.value) for p in parsed]
        prediction = self._distribution(question, pairs, lower, upper)
        readable = ", ".join(f"p{round(p * 100)}={_fmt(v)}" for p, v in pairs)
        self._note_run(question, model, readable, parsed_by)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ------------------------------------------------------------------ date

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning, model = await self._invoke_forecaster(prompt)
        lower, upper = question.lower_bound.timestamp(), question.upper_bound.timestamp()
        pairs = parsing.parse_date_percentiles(reasoning)
        parsed_by = "regex"
        if pairs is None:
            parsed_by = "llm"
            instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a date question.
                - This text is trying to answer the question: "{question.question_text}".
                {self._create_single_distribution_parsing_message(question)}
                {self._create_resolved_question_parsing_message()}
                - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
                - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                """
            )
            parsed: list[DatePercentile] = await structure_output(
                reasoning,
                list[DatePercentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=instructions,
                num_validation_samples=self._structure_output_validation_samples,
            )
            pairs = [(p.percentile, _as_utc(p.value).timestamp()) for p in parsed]
        prediction = self._distribution(question, pairs, lower, upper)
        readable = ", ".join(
            f"p{round(p * 100)}={datetime.fromtimestamp(v, tz=timezone.utc):%Y-%m-%d}" for p, v in pairs
        )
        self._note_run(question, model, readable, parsed_by)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _distribution(
        self,
        question: NumericQuestion | DateQuestion,
        pairs: list[tuple[float, float]],
        lower: float,
        upper: float,
    ) -> NumericDistribution:
        clean = numeric.sanitize_percentiles(
            pairs, lower_bound=lower, upper_bound=upper, zero_point=question.zero_point
        )
        prediction = NumericDistribution.from_question(
            [Percentile(percentile=p, value=v) for p, v in clean], question
        )
        # Build the CDF now, so a run whose distribution cannot be turned into a valid CDF fails
        # here (and is dropped from the ensemble) instead of breaking the aggregation later.
        heights = [point.percentile for point in cdf_points(prediction)]
        problems = numeric.validate_cdf(
            heights,
            cdf_size=question.cdf_size,
            open_lower_bound=question.open_lower_bound,
            open_upper_bound=question.open_upper_bound,
        )
        if problems:
            raise ValueError(f"run produced an invalid CDF: {problems}")
        return prediction

    # ------------------------------------------------------------------ aggregation

    async def _aggregate_predictions(self, predictions: list, question: MetaculusQuestion):
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        calibration = self.cfg.calibration
        if isinstance(question, BinaryQuestion):
            return aggregate.aggregate_binary(
                predictions, low=calibration.binary_min, high=calibration.binary_max
            )
        if isinstance(question, MultipleChoiceQuestion):
            options = list(question.options)
            runs = [{o.option_name: o.probability for o in p.predicted_options} for p in predictions]
            final = aggregate.aggregate_options(runs, options, floor=calibration.option_min)
            return _option_list(options, final)
        if isinstance(question, (NumericQuestion, DateQuestion)):
            return self._aggregate_numeric(predictions, question)
        raise UnsupportedQuestion(f"unsupported question type {type(question).__name__}")

    def _aggregate_numeric(
        self, predictions: list[NumericDistribution], question: NumericQuestion | DateQuestion
    ) -> NumericDistribution:
        rules = dict(
            cdf_size=question.cdf_size,
            open_lower_bound=question.open_lower_bound,
            open_upper_bound=question.open_upper_bound,
        )
        cdfs: list[list[float]] = []
        x_axis: list[float] | None = None
        for prediction in predictions:
            points = cdf_points(prediction)
            heights = [point.percentile for point in points]
            if numeric.validate_cdf(heights, **rules):
                continue
            cdfs.append(heights)
            x_axis = [point.value for point in points]
        if not cdfs or x_axis is None:
            raise ValueError("no valid CDF among the runs")
        try:
            final = numeric.ensure_valid_cdf(numeric.mixture_cdf(cdfs), **rules)
        except ValueError as error:
            logger.warning("mixture CDF rejected (%s); falling back to the pointwise median", error)
            import numpy as np

            final = numeric.ensure_valid_cdf(np.median(np.asarray(cdfs), axis=0).tolist(), **rules)
        self.final_cdfs[question_key(question)] = final
        first = predictions[0]
        return NumericDistribution(
            declared_percentiles=[Percentile(value=x, percentile=h) for x, h in zip(x_axis, final)],
            open_upper_bound=first.open_upper_bound,
            open_lower_bound=first.open_lower_bound,
            upper_bound=first.upper_bound,
            lower_bound=first.lower_bound,
            zero_point=first.zero_point,
            cdf_size=question.cdf_size,
            standardize_cdf=False,
            strict_validation=False,
            is_date=first.is_date,
        )

    # ------------------------------------------------------------------ comment

    def _create_unified_explanation(self, question, research_prediction_collections, aggregated_prediction, final_cost, time_spent_in_minutes) -> str:
        explanation = super()._create_unified_explanation(
            question, research_prediction_collections, aggregated_prediction, final_cost, time_spent_in_minutes
        )
        runs = self.run_details.get(question_key(question), [])
        lines = [f"- {run['model']}: {run['prediction']}" for run in runs]
        notes = (
            "\n\n# NOTES\n"
            "Automated forecast, no human in the loop. Research from several web-search sources, "
            "then independent runs on models from different labs; binary = median of runs, "
            "multiple choice = mean, numeric = equal-weight mixture of the runs' distributions.\n"
            + "\n".join(lines)
        )
        return explanation + notes


def cdf_points(prediction: NumericDistribution) -> list[Percentile]:
    """The library's CDF for a distribution, with a time limit (see bot/guard.py)."""
    with time_limit(CDF_TIME_LIMIT_S, "building a CDF"):
        return prediction.get_cdf()


def _option_list(options: list[str], probabilities: dict[str, float]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=o, probability=probabilities[o]) for o in options]
    )


def _plausible(pairs: list[tuple[float, float]], lower: float, upper: float) -> bool:
    span = upper - lower
    return any(lower - span <= value <= upper + span for _, value in pairs)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _fmt(value: float) -> str:
    return f"{value:.6g}"
