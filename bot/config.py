"""Typed view of config.toml. No secrets here."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"


@dataclass(frozen=True)
class Profile:
    name: str
    forecasters: tuple[str, ...]
    search_models: tuple[str, ...]
    use_asknews: bool = False


@dataclass(frozen=True)
class Tournament:
    name: str
    id: str | int
    profile: str
    min_season_budget_left_usd: float = 0.0


@dataclass(frozen=True)
class Models:
    parser: str
    reasoning_effort: str | None = "high"
    forecast_timeout_s: int = 240
    search_timeout_s: int = 180
    parser_timeout_s: int = 60
    allowed_tries: int = 2
    max_output_tokens: int = 20000
    # Tries for particular models (price-table name -> tries), overriding allowed_tries.
    allowed_tries_by_model: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Ensemble:
    min_successful_fraction: float = 0.4
    model_failover: int = 1
    allow_forecast_without_research: bool = False
    # "per_run" (binary median of runs) or "per_model_mean" (one vote per model); see bot/aggregate.py.
    # The defaults are the original behaviour, so a config.toml without the key runs as before.
    aggregate: str = "per_run"


@dataclass(frozen=True)
class Calibration:
    binary_min: float = 0.03
    binary_max: float = 0.97
    option_min: float = 0.02
    binary_extremize: float = 1.0   # logit(p) x this after aggregation, before the clip; 1 = off


@dataclass(frozen=True)
class Limits:
    max_question_usd: float = 1.5
    max_run_usd: float = 6.0
    max_day_usd: float = 12.0
    max_season_usd: float = 400.0
    max_llm_calls_per_question: int = 24
    openrouter_min_remaining_usd: float = 2.0
    max_attempts_per_question: int = 6
    max_concurrent_questions: int = 2
    question_timeout_s: int = 600
    run_time_budget_s: int = 780
    min_seconds_before_close: int = 360
    asknews_monthly_call_units: int = 900
    asknews_units_per_question: int = 6
    research_chars_per_source: int = 12000


@dataclass(frozen=True)
class Prices:
    per_million: dict[str, tuple[float, float]] = field(default_factory=dict)
    default: tuple[float, float] = (15.0, 75.0)
    search_call_fee_usd: float = 0.04

    def for_model(self, model: str) -> tuple[float, float]:
        return self.per_million.get(normalise_model_name(model), self.default)


@dataclass(frozen=True)
class Config:
    armed: bool
    state_dir: Path
    log_dir: Path
    tournaments: tuple[Tournament, ...]
    testing: Tournament
    profiles: dict[str, Profile]
    models: Models
    ensemble: Ensemble
    calibration: Calibration
    limits: Limits
    prices: Prices

    def profile(self, name: str) -> Profile:
        return self.profiles[name]

    def tournament(self, name_or_id: str) -> Tournament:
        for t in (*self.tournaments, self.testing):
            if name_or_id in (t.name, str(t.id)):
                return t
        raise KeyError(f"unknown tournament {name_or_id!r}")


def normalise_model_name(model: str) -> str:
    """'openrouter/openai/gpt-x:online' -> 'openai/gpt-x' (key used in the price table)."""
    name = model.removeprefix("openrouter/").removeprefix("metaculus/")
    return name.split(":", 1)[0]


def is_search_model(model: str) -> bool:
    return ":online" in model or "search" in model


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    base = path.resolve().parent

    profiles = {
        name: Profile(
            name=name,
            forecasters=tuple(body["forecasters"]),
            search_models=tuple(body.get("search_models", [])),
            use_asknews=bool(body.get("use_asknews", False)),
        )
        for name, body in raw.get("profile", {}).items()
    }
    tournaments = tuple(Tournament(**t) for t in raw.get("tournament", []))
    testing_raw = raw.get("testing", {"id": "bot-testing-area", "profile": "full"})
    testing = Tournament(name="testing", **testing_raw)

    for t in (*tournaments, testing):
        if t.profile not in profiles:
            raise ValueError(f"tournament {t.name!r} uses unknown profile {t.profile!r}")
    for p in profiles.values():
        if not p.forecasters:
            raise ValueError(f"profile {p.name!r} has no forecasters")

    prices_raw = dict(raw.get("prices", {}))
    default_price = tuple(prices_raw.pop("default", (15.0, 75.0)))
    search_fee = float(prices_raw.pop("search_call_fee_usd", 0.04))
    prices = Prices(
        per_million={k: (float(v[0]), float(v[1])) for k, v in prices_raw.items()},
        default=(float(default_price[0]), float(default_price[1])),
        search_call_fee_usd=search_fee,
    )

    calibration = Calibration(**raw.get("calibration", {}))
    if not (0.001 <= calibration.binary_min < calibration.binary_max <= 0.999):
        raise ValueError("calibration.binary_min/max must satisfy 0.001 <= min < max <= 0.999")
    if not (0.5 <= float(calibration.binary_extremize) <= 3.0):
        raise ValueError("calibration.binary_extremize must be within [0.5, 3.0] (1.0 = off)")

    ensemble = Ensemble(**raw.get("ensemble", {}))
    if ensemble.aggregate not in ("per_run", "per_model_mean"):
        raise ValueError("ensemble.aggregate must be \"per_run\" or \"per_model_mean\"")
    if ensemble.model_failover < 0:
        raise ValueError("ensemble.model_failover must be >= 0")

    return Config(
        armed=bool(raw.get("armed", False)),
        state_dir=(base / raw.get("state_dir", "state")),
        log_dir=(base / raw.get("log_dir", "logs")),
        tournaments=tournaments,
        testing=testing,
        profiles=profiles,
        models=Models(**raw.get("models", {})),
        ensemble=ensemble,
        calibration=calibration,
        limits=Limits(**raw.get("limits", {})),
        prices=prices,
    )
