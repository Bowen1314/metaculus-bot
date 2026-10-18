# Metaculus FutureEval bot (Fall 2026)

An unattended forecasting bot for the Fall 2026 FutureEval Bot Tournament ($50,000 pool) and the
two-week MiniBench rounds ($1,000 each). Built on the official `forecasting-tools` library and its
`FallTemplateBot2026`; the changes on top of the template and the evidence for each are in
[RESEARCH.md](RESEARCH.md), section 4.

State on 2026-10-04: the bot account `13owenbot` runs on a small always-on server with the
Metaculus-donated OpenRouter credits. 213 offline tests pass; dry runs on the unscored testing
area produce valid forecasts for binary, multiple-choice and numeric questions. The bot is fully
automated: no human looks at or edits its forecasts on open tournament questions.

The tournament started on 2026-09-28 and question windows are about 3 hours, so the bot runs
every 15 minutes (a skipped question scores 0, it is not a penalty).

## What you need to do, in order

1. **Fill in the participation form**: https://forms.gle/aQdYMq9Pisrf1v7d8
   The first section is mandatory for every participant (email, primary bot username
   `13owenbot`, "Who are you?"). Answer **Yes** to "Do you want to apply for free LLM credits?".
   The credits come from OpenAI, Anthropic and Google and are delivered as an **OpenRouter key**.
   Amount and waiting time are not published.
2. **Optional, free: AskNews.** Email contact@asknews.app with the bot name, the email you
   registered on Metaculus, your name, LinkedIn and affiliation. One account per person,
   1,000 calls a month. The bot works without it (it then uses two web-search models only).
3. **When the OpenRouter key arrives**, add it to `.env` in this directory, next to the token:
   ```
   OPENROUTER_API_KEY=...
   ```
   (and `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` if you have them; `env.example` lists every name
   the bot accepts). Keep the file at mode 600. Then set `max_season_usd` in `config.toml` to the
   size of the grant.
4. **Check the set-up** (read-only, prints names and statuses, never secret values):
   ```
   .venv/bin/python run.py doctor
   ```
   Every line should be `ok` (it also compares the model ids and the price table in
   `config.toml` with OpenRouter's public model list).
5. **Dry run with the real models**, on the unscored bot testing area. Nothing is submitted.
   Expect roughly $0.50 per question:
   ```
   .venv/bin/python run.py dry-run --limit 3 --verbose
   ```
   Read the output: are the probabilities sensible, do the numeric ranges make sense, is the
   comment readable. This is the review that has to happen before going live. Change prompts or
   models now if needed; after going live the rules forbid looking at the bot's output on open
   tournament questions and then changing it.
6. **Optional: one real submission to the testing area** (unscored, no prize pool). Posts one
   forecast and one private comment under the bot's name, to prove the write path works:
   ```
   .venv/bin/python run.py submit-test --confirm
   ```
   This has not been run by the build agent; it is your call.
7. **Arm it**: in `config.toml` change `armed = false` to `armed = true`.
8. **Put it on the server** (next section) and enable the timer. From then on it needs no
   attention; see "During the season".
9. **At the end of the season, to be paid**: answer the bot-maker survey Metaculus emails you,
   give them the code or a description (this repository), and pass their identity / residency
   check. Unclaimed prizes can be forfeited after 30 days. Payment is by bank transfer (Ramp).

Nothing in this list requires paying for anything.

## Running it on the server (recommended)

Needs Docker and about 300 MB of RAM while a run is in progress. Not done by the build agent.

```
# on the Mac: copy the project including .env, without the local venv and state
rsync -a --exclude .venv --exclude state --exclude logs --exclude .git \
      ./ SERVER:/opt/metaculus-bot/      # run from the project directory

# on the server
cd /opt/metaculus-bot
chmod 600 .env
docker build -t metaculus-bot .
deploy/run-once.sh doctor                 # same check as step 4, inside the container
cp deploy/metaculus-bot.service deploy/metaculus-bot.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now metaculus-bot.timer
```

`deploy/run-once.sh` is the exact run command; written out it is:

```
docker run --rm --user "$(id -u):$(id -g)" --memory 700m --cpus 1 \
  -v /opt/metaculus-bot/.env:/app/.env:ro \
  -v /opt/metaculus-bot/config.toml:/app/config.toml:ro \
  -v /opt/metaculus-bot/state:/app/state \
  -v /opt/metaculus-bot/logs:/app/logs \
  metaculus-bot run --live
```

The timer starts it every 15 minutes. A run that is still going when the next one starts is
left alone (file lock in `state/`); the newcomer exits at once. `state/` holds the ledger and
must be kept: it is what stops double submissions and what remembers the spend.

Without Docker: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`, and have
the timer call `.venv/bin/python run.py run --live` from the project directory.

**GitHub Actions** is what the official template uses. `deploy/github-actions-example.yml` shows
what it would need (a GitHub repository, the secrets, the state directory kept in the Actions
cache). It is not recommended here: scheduled workflows start late or get skipped, they are
switched off after 60 days without commits, and the spend totals live in a cache that can vanish.
A server that is on anyway has none of these problems.

## During the season

```
deploy/run-once.sh status        # or: .venv/bin/python run.py status
tail -f logs/bot.log
```

- `status` shows questions by state and money spent. `submitted`/`done` are finished;
  `failed` will be retried on later runs (up to 6 times while the question is open);
  `gave_up` means it could not be forecast in time.
- The bot stops by itself when a limit in `config.toml` is hit: $1.50 per question, $6 per run,
  $12 per day, `max_season_usd` in total, or less than $2 left on the OpenRouter key.
- To pause: `systemctl stop metaculus-bot.timer`, or set `armed = false`.
- Every new season: change the tournament id in `config.toml` (`fall-futureeval-2026`) and
  review the model ids. `minibench` always points at the current round.
- Do not re-run a question by hand and do not delete rows from the ledger: one forecast per
  question, made without a human looking at it first, is a tournament rule.

## What is not verified yet

Needs the OpenRouter key:
- an actual model call. The model ids and prices in `config.toml` were checked against
  OpenRouter's public list on 2026-10-02 (`doctor` repeats that check), but no call has been
  made, so the `reasoning_effort = "high"` setting and the `:online` web search are untested;
- whether 20,000 output tokens per call (`max_output_tokens`) is enough at high reasoning
  effort; if the dry run shows "empty answer" warnings, raise it;
- the real cost per question (estimate: about $0.50 on the full profile, $0.25 on the lean one)
  and whether OpenRouter reports the cost of web searches; the bot charges itself the larger
  of the reported cost and its own price table;
- whether the key reports its remaining credit (if not, only the local limits apply);
- forecast quality. All tests use stand-in answers; no real model has been called.

Needs your go-ahead:
- an actual POST of a forecast and a comment. The payloads are checked against a copy of
  Metaculus's own validation code, but nothing has ever been sent (step 6).

Needs AskNews credentials:
- the AskNews path against the real service, and how AskNews counts calls (the bot assumes
  6 units per question and stops using it at 900 units in a month).

Other:
- the Docker image has not been built (no Docker daemon on the build machine). All 81 pinned
  packages were checked to have Linux wheels for Python 3.13 on x86_64 and arm64.
- conditional questions ("if A then B") are skipped; none have been seen in this tournament.
- the bot account's API tier is shown as `restricted`; reads work, write limits are unknown
  until the first submission.

## What to expect

Details and sources are in RESEARCH.md. In short: prizes are proportional to the square of the
total peer score, and anything under $50 is not paid, so a season ends with either nothing or
at least about $50. In the last three seasons the plain template with a current flagship model
would have earned $120 to $1,450, and 75-88% of bots that answered at least 90% of questions
were paid. A reasonable expectation for this bot, if credits arrive within a week or two and
it then runs to the end of the season: roughly a 55-65% chance of any prize, a few hundred
dollars if so (expected value around $300-400), and a 35-45% chance of nothing. MiniBench adds
perhaps $20 per two-week round on average. Without donated credits the model calls would cost
about $200 for the season plus about $15 per MiniBench round, which is roughly what the bot is
expected to win, so the plan only makes sense with the credits.

## Layout

```
run.py              command line: doctor, status, dry-run, submit-test, run --live
config.toml         everything that is not a secret: tournaments, models, limits, prices, armed
env.example         the names .env may contain
bot/envguard.py     loads .env, removes every credential from the ambient environment
bot/forecaster.py   EnsembleBot on top of the library's FallTemplateBot2026
bot/research.py     web-search models (+ AskNews), each allowed to fail
bot/parsing.py      reads the models' answers without an LLM where possible
bot/numeric.py      percentile clean-up, Metaculus's CDF rules, repair, mixture
bot/aggregate.py    one vote per model: mean + extremize + clip (binary), mean + floor (multiple choice)
bot/runner.py       one pass: select, forecast, submit, recover
bot/ledger.py       SQLite record of every question, spend and run
bot/spend.py        per-question / run / day / season limits
bot/lock.py, guard.py   one run at a time; time limits for code that can hang
tests/              209 offline tests:  .venv/bin/python -m pytest -q
deploy/             run-once.sh, systemd service + timer, GitHub Actions example
RESEARCH.md         rules, payout history, evidence behind the design
PROGRESS.md         build log
```
