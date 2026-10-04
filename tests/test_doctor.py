"""The model check in `doctor` (the rest of doctor talks to the network and is not unit-tested)."""

from __future__ import annotations

import dataclasses

from bot.config import normalise_model_name
from bot.doctor import check_models


def catalog_for(config, *, price_factor=1.0, parameters=("reasoning", "max_tokens")) -> dict:
    names = {normalise_model_name(config.models.parser)}
    for profile in config.profiles.values():
        names |= {normalise_model_name(m) for m in (*profile.forecasters, *profile.search_models)}
    catalog = {}
    for name in names:
        prompt, completion = config.prices.for_model(name)
        catalog[name] = {
            "id": name,
            "pricing": {"prompt": str(prompt * price_factor / 1e6), "completion": str(completion * price_factor / 1e6)},
            "supported_parameters": list(parameters),
        }
    return catalog


def test_all_configured_models_listed_and_priced(config):
    lines = check_models(config, catalog_for(config))
    assert lines and {status for status, _ in lines} == {"ok"}


def test_unknown_model_id_fails(config):
    catalog = catalog_for(config)
    del catalog["openai/gpt-6.1-sol"]
    failures = [text for status, text in check_models(config, catalog) if status == "FAIL"]
    assert failures and all("openai/gpt-6.1-sol" in text for text in failures)


def test_price_table_below_list_price_and_missing_reasoning_are_flagged(config):
    lines = check_models(config, catalog_for(config, price_factor=2.0, parameters=("max_tokens",)))
    text = "\n".join(t for status, t in lines if status == "warn")
    assert "price table is too low" in text and "reasoning_effort will be ignored" in text
    assert not [s for s, _ in lines if s == "FAIL"]


def test_models_called_directly_are_not_checked_against_openrouter(config):
    direct = dataclasses.replace(config.profile("lean"), forecasters=("anthropic/claude-opus-5.5",), search_models=())
    cfg = dataclasses.replace(config, profiles={"lean": direct}, models=dataclasses.replace(config.models, parser="gpt-6-luna"))
    assert check_models(cfg, {}) == []
