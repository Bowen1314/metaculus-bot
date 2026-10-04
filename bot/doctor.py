"""`python run.py doctor`: read-only checks. Prints names and statuses, never a secret value."""

from __future__ import annotations

import os
from typing import Any

import requests

from bot.config import Config, normalise_model_name
from bot.envguard import EnvReport
from bot.runner import missing_credentials, openrouter_remaining_usd

API = "https://www.metaculus.com/api"


def _get(path: str, token: str, **params: Any) -> Any:
    response = requests.get(
        f"{API}{path}",
        headers={"Authorization": f"Token {token}", "Accept-Language": "en"},
        params=params or None,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def check_models(config: Config, catalog: dict[str, dict]) -> list[tuple[str, str]]:
    """Compares the configured OpenRouter models with OpenRouter's public model list.
    Returns (status, text) lines. No credentials are involved."""
    lines: list[tuple[str, str]] = []
    roles: dict[str, set[str]] = {}
    roles.setdefault(config.models.parser, set()).add("parser")
    for profile in config.profiles.values():
        for model in profile.forecasters:
            roles.setdefault(model, set()).add("forecast")
        for model in profile.search_models:
            roles.setdefault(model, set()).add("search")
    for model in sorted(roles):
        if not model.startswith("openrouter/"):
            continue
        name = normalise_model_name(model)
        entry = catalog.get(name)
        if entry is None:
            lines.append(("FAIL", f"model '{name}' is not in OpenRouter's model list; fix the id in config.toml"))
            continue
        notes = []
        try:
            pricing = entry.get("pricing") or {}
            listed = (float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6)
            ours = config.prices.for_model(model)
            if ours[0] + 1e-9 < listed[0] or ours[1] + 1e-9 < listed[1]:
                notes.append(
                    f"price table is too low (config {ours[0]:g}/{ours[1]:g}, OpenRouter {listed[0]:g}/{listed[1]:g} "
                    "USD per million tokens); raise it in [prices] or the spend guard undercounts"
                )
        except (KeyError, TypeError, ValueError):
            notes.append("no price listed")
        supported = entry.get("supported_parameters") or []
        if "forecast" in roles[model] and config.models.reasoning_effort and supported and "reasoning" not in supported:
            notes.append("does not list the 'reasoning' parameter (reasoning_effort will be ignored)")
        lines.append(("warn" if notes else "ok", f"model '{name}' ({', '.join(sorted(roles[model]))})" + (": " + "; ".join(notes) if notes else " is listed on OpenRouter")))
    return lines


def _openrouter_catalog() -> dict[str, dict]:
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)  # public, sent without any key
    response.raise_for_status()
    return {entry["id"]: entry for entry in response.json().get("data", []) if "id" in entry}


def doctor(config: Config, report: EnvReport) -> int:
    ok = True

    def line(status: str, text: str) -> None:
        print(f"[{status:^5}] {text}")

    line("info", report.describe())
    for warning in report.warnings:
        line("warn", warning)
    line("info", f"armed = {config.armed} (live submission to scored tournaments is {'ON' if config.armed else 'off'})")

    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        line("FAIL", "METACULUS_TOKEN is not in .env")
        ok = False
    else:
        try:
            me = _get("/users/me/", token)
            line("ok", f"Metaculus auth works: username={me.get('username')} id={me.get('id')} is_bot={me.get('is_bot')}")
            if not me.get("is_bot"):
                line("FAIL", "this token belongs to a human account, not a bot account; create a bot in Settings > My forecasting bots")
                ok = False
        except Exception as error:
            line("FAIL", f"Metaculus auth check failed: {error.__class__.__name__}")
            ok = False
        if ok:
            for tournament in (*config.tournaments, config.testing):
                try:
                    project = _get(f"/projects/tournaments/{tournament.id}/", token)
                    posts = _get("/posts/", token, tournaments=str(tournament.id), statuses="open", limit=100)
                    results = [post for post in posts.get("results", []) if "notebook" not in post]
                    shown = len(results)
                    open_now = f"{shown}+" if len(posts.get("results", [])) >= 100 else str(shown)
                    line(
                        "ok",
                        f"{tournament.name}: '{project.get('name')}' (project {project.get('id')}), "
                        f"prize pool {project.get('prize_pool')}, open question posts now: {open_now}",
                    )
                except Exception as error:
                    line("warn", f"{tournament.name}: could not read tournament '{tournament.id}': {error.__class__.__name__}")

    models: list[str] = [config.models.parser]
    for profile in config.profiles.values():
        models += [*profile.forecasters, *profile.search_models]
    if any(model.startswith("openrouter/") for model in models):
        try:
            for status, text in check_models(config, _openrouter_catalog()):
                line(status, text)
                ok = ok and status != "FAIL"
        except Exception as error:
            line("warn", f"could not read OpenRouter's model list: {error.__class__.__name__}")
    missing = missing_credentials(models)
    if missing:
        line("FAIL", f"the configured models need these in .env: {', '.join(missing)} (forecasting is blocked until then)")
        ok = False
    else:
        line("ok", "provider credentials for the configured models are present")
        remaining = openrouter_remaining_usd()
        if remaining is None:
            line("info", "OpenRouter credit: not reported (no limit on the key, or the check failed)")
        else:
            status = "ok" if remaining >= config.limits.openrouter_min_remaining_usd else "FAIL"
            ok = ok and status == "ok"
            line(status, f"OpenRouter credit left: ${remaining:.2f}")

    has_asknews = bool(
        (os.environ.get("ASKNEWS_CLIENT_ID") and os.environ.get("ASKNEWS_SECRET")) or os.environ.get("ASKNEWS_API_KEY")
    )
    line("ok" if has_asknews else "info", "AskNews credentials: " + ("present" if has_asknews else "not set (optional; web search is used alone)"))

    try:
        config.state_dir.mkdir(parents=True, exist_ok=True)
        probe = config.state_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        line("ok", f"state directory writable: {config.state_dir}")
    except OSError as error:
        line("FAIL", f"state directory not writable: {error.__class__.__name__}")
        ok = False

    print("\nready to forecast" if ok else "\nNOT ready: fix the FAIL lines above")
    return 0 if ok else 3
