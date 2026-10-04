# Research notes: rules, payouts, what works

Collected 2026-10-02. Sources are listed per section. Where something is marked "(API)" it was read from the
Metaculus API with the bot token (read-only GETs); "(code)" means the open-source Metaculus site repo at
commit `7ac38781` (2026-10-02).

## 1. Rules that constrain us

Sources: resources page https://www.metaculus.com/notebooks/38928/futureeval-resources-page/ ,
official rules https://www.metaculus.com/notebooks/28751/q1-ai-forecasting-benchmark-rules/ ,
participation form https://forms.gle/aQdYMq9Pisrf1v7d8 .

| Topic | Rule | Consequence for us |
|---|---|---|
| Human in the loop | "A bot maker cannot preview how their bot forecasts on open or upcoming questions and then update their bot based on that preview." Also: "a bot maker cannot decide that they don't like their bot's forecast and rerun the bot on that question." | Never dry-run on open tournament questions and then change the bot. Review dry runs on `bot-testing-area` only. One forecast per question, enforced by the ledger. No manual re-runs. |
| Comments | Bot must leave a comment (with its forecast/reasoning) on every question it forecasts; "bots should only leave private comments". Needed for prize eligibility. | Runner posts a private comment after each forecast and retries it separately if it fails. |
| One bot per person | "There can only be one 'prize-winning' bot per participant/team"; extra bots must be labelled secondary. Rules page: "An individual may enter only a single bot". | One bot. API shows `13owenbot` has `is_primary_bot: true`. |
| Commercial bots | "any bot that is built or operated on behalf of a for-profit entity that has three or more founders/employees/contractors" - no prize money (unless fully open-sourced by season end), no LLM credits. Solo/personal projects are not commercial. | A personal project qualifies. Do not run it on behalf of a company. |
| Copying the community prediction | Bots may not blindly copy the CP. | We never read the CP. |
| Eligibility | 18+; not resident of Cuba, Iran, Syria, North Korea, Russia, Crimea/Donetsk/Luhansk; not Metaculus staff/family. Winners "will be required to verify their identity, nationality, and residency". | Owner must be able to pass that check. |
| Prize conditions | Provide the code or a description of how the bot works; agree to a code inspection; return the bot-maker survey ("Prizes are only distributed after all bot maker surveys have been received for the season"). Unclaimed after 30 days may be forfeited. Taxes/fees on recipient. | Keep the repo tidy; answer Metaculus emails within 30 days. |
| Payment | Bank transfer through Ramp; country must be on Ramp's supported list. | Owner checks country. |
| Participation form | First section is mandatory for all participants: 3 required questions - email, primary bot username, "Who are you?" (e.g. Student Hobbyist). Optional: why participating, repo link, "Do you want to apply for free LLM credits? (Yes/No)". | Owner fills it once; tick Yes for credits. |
| LLM credits | Donated by OpenAI, Anthropic, Google, delivered through OpenRouter. "Credits must be used for ... running bots ... in our tournament and related activities." No guaranteed top-up if exhausted. Amount and turnaround are not published. | Bot uses `OPENROUTER_API_KEY`; only OpenAI/Anthropic/Google models; spend guard. |
| Metaculus LLM proxy | Tested with the bot token: `"You don't have an allowance for model <gpt-4o-mini> on <Openai>"`. | The token alone gives no LLM access. An OpenRouter key is required before the bot can forecast. |
| AskNews | Free for registered bot builders: "1k calls per month, 4k calls total for the tournament", one account per person. Latest-news search = 1 call, archive search = 5 calls. Request by emailing contact@asknews.app (bot name, registered email, name, LinkedIn, affiliation). | Optional. If present, used for the seasonal tournament only (6 call-units/question), with a monthly cap in the ledger. |
| Other search tools | Exa, Perplexity etc. are not covered by donated credits. OpenRouter `:online` (native provider search) is covered. | Research uses `:online` on OpenAI and Google models. |
| Bot testing area | "you can test whether it is working by running it on questions in the bot testing area, which won't be scored." (API): no prize pool, `bot_leaderboard_status: exclude_and_show`. | Safe place for dry runs and one real submission test. |

Tournament facts (API, 2026-10-02):

- `fall-futureeval-2026`: id 33121, pool $50,000, `spot_peer_tournament`, started 2026-09-28, forecasting ends 2027-01-06,
  closes 2027-03-05. 11 questions so far, **316 forecasters already**.
- `minibench`: currently points at the round starting 2026-10-05 (id 33129), pool $1,000, 0 questions yet.
  Previous rounds have their own slugs (`minibench-2026-09-21`, ...), ~60 questions each.
- Observed windows: `open_time` to `scheduled_close_time` is 3.0 h on the questions seen (documentation says 1.5 h);
  MiniBench's 60 questions mostly open within the first day or two of a round. A 15-minute timer is sufficient.
- Question mix seen: binary, multiple choice, numeric (201-point CDF), discrete (CDF sizes 8 to 162).
- The bot account: `13owenbot`, `is_bot: true`, `api_forecasting_access: enabled`, `api_access_tier: restricted`.

## 2. How prize money is computed

Source (code): `scoring/utils.py`, `scoring/models.py`, `scoring/score_math.py`.

- Per question: spot peer score = `100 * n/(n-1) * ln(p_you / geometric_mean_of_all)` on the outcome that happened
  (halved for numeric/discrete, where `p` is your probability mass in the bucket containing the answer).
  A question you skip scores 0.
- Tournament: `score = sum(question_score * question_weight)`; `take = max(score, 0) ** 2`;
  prize share = take / sum of takes.
- **Minimum prize**: `minimum_prize_amount` defaults to $50. Anyone whose share would be below it gets nothing and the
  money is redistributed upward. So payouts are either $0 or at least about $50 - there are no "tens of dollars" prizes.
- Metaculus's own `metac-*` template bots are on the leaderboard and count as peers but are excluded from prizes.

## 3. What past seasons actually paid (API leaderboards)

Non-Metaculus, prize-eligible entries only.

| Season | Pool | Eligible bots | Positive score | Paid | Top | Median paid | Smallest |
|---|---|---|---|---|---|---|---|
| Fall 2025 | $50k | 71 | 38 | 31 | $6,859 | $972 | $63 |
| Spring 2026 | $50k | 102 | 53 | 30 | $5,097 | $1,502 | $54 |
| Summer 2026 (not finalized) | $50k | 192 | 92 | 41 | $3,392 | $854 | $55 |

Full prize lists:

- Fall 2025: 6859, 6009, 5793, 3581, 3316, 2643, 2518, 1942, 1763, 1564, 1552, 1526, 1496, 1285, 996, 972, 923, 789,
  774, 697, 571, 500, 481, 421, 255, 188, 175, 155, 107, 85, 63
- Spring 2026: 5097, 4767, 4186, 3604, 2816, 2720, 2635, 2524, 2503, 2203, 1959, 1939, 1743, 1606, 1535, 1468, 1117,
  1053, 976, 769, 664, 490, 337, 309, 247, 195, 183, 180, 120, 54
- Summer 2026: 3392, 3370, 3105, 2946, 2799, 2740, 2359, 2333, 2123, 2121, 2090, 2053, 1932, 1909, 1862, 1178, 1094,
  1089, 898, 873, 854, 833, 799, 753, 728, 599, 428, 349, 339, 318, 315, 298, 264, 193, 174, 150, 93, 78, 58, 56, 55

Coverage is the dividing line:

| Season | Coverage >= 90%: n / paid / median / mean | Coverage 50-90%: n / paid / median / mean | Coverage < 50%: n / paid |
|---|---|---|---|
| Fall 2025 | 17 / 15 (88%) / $1,526 / $2,098 | 31 / 15 (48%) / $0 / $458 | 23 / 1 |
| Spring 2026 | 16 / 12 (75%) / $2,353 / $2,046 | 33 / 14 (42%) / $0 / $481 | 53 / 4 |
| Summer 2026 | 30 / 25 (83%) / $1,136 / $1,437 | 29 / 7 (24%) / $0 / $161 | 133 / 9 |

Caveat: the >=90% group is self-selected (experienced teams that ran all season), so it overstates what a first-season
bot gets. The better reference class is the next table.

### How the plain official template ranked

The `metac-*` bots are the official template (one search, one prompt, five runs, median) with different models.
"Would-be prize" is what that score would have earned had the bot been eligible.

| Season | Template bot (model) | Overall rank | Avg peer score/question | Would-be prize |
|---|---|---|---|---|
| Fall 2025 | gpt-5 | 16 of 136 | 9.9 | $1,450 |
| | gpt-5-high | 17 | 9.8 | $1,397 |
| | claude-4-1-opus-high | 26 | 6.3 | $602 |
| | gemini-2-5-pro | 37 | 4.3 | $286 |
| Spring 2026 | gpt-5-1-high | 18 of 180 | 10.0 | $1,269 |
| | claude-4-5-sonnet-high | 23 | 7.3 | $700 |
| | gpt-5-2-high | 26 | 6.9 | $622 |
| | gpt-5-2 | 34 | 4.8 | $306 |
| | claude-opus-4-5-high | 45 | 3.3 | $145 |
| | gpt-5-1 | 47 | 3.0 | $121 |
| Summer 2026 | claude-opus-4-8-high | 24 of 284 | 9.8 | $842 |
| | gemini-3-5-flash | 30 | 8.3 | $636 |
| | gpt-5-5-instant | 33 | 7.8 | $576 |
| | gpt-5-5-high | 35 | 7.3 | $509 |
| | claude-sonnet-4-6 | 37 | 6.5 | $402 |
| | claude-opus-4-7-high | 38 | 6.4 | $385 |
| | gpt-5-4-high | 41 | 5.9 | $329 |
| | gemini-3-1-pro-high | 45 | 5.4 | $284 |
| | gemini-3-1-pro | 56 | 3.7 | $136 |

Template bots with old or small models are deeply negative (gpt-4o-mini, gpt-3.5: -13 to -25 per question). The
template's built-in default model is still gpt-4o, so running it unmodified would lose.

Dollar value of a given average score at full coverage: avg 5 = $431 / $328 / $248 (Fall 25 / Spring 26 / Summer 26);
avg 7 = $837 / $639 / $484; avg 10 = $1,679 / $1,287 / $979. The same score buys less each season as the field grows.

### MiniBench (API, six finalized rounds, Jun-Sep 2026)

$1,000 per round, 68-114 eligible bots, 8-13 paid, prizes $51-$196 (median $68-$123). Among bots with >=90% coverage,
20-38% were paid and the mean take was $18-$32 per round.

Other published numbers (not from the API): Q4 2024 top five $9,658 / $4,477 / $3,930 / $3,410 / $3,312 of $30k
(https://www.lesswrong.com/posts/sLsweqaHKJyMPRv68/); Q2 2025 top three $7,550 / $4,563 / $3,270 and "16 other bot
competitors won a total of $14,617" (https://www.metaculus.com/notebooks/39140/). Spring 2026 survey says "37 of 133
participating owners" won prizes; the API leaderboard shows 30 paid entries - the difference is unexplained.

## 4. What the better bots did, and what we took from it

Sources: Q4 2024 results https://www.metaculus.com/notebooks/35291/ ; Q1 2025 https://www.metaculus.com/notebooks/38673/ ;
Q2 2025 https://www.metaculus.com/notebooks/40456/ ; Fall 2025 survey https://www.metaculus.com/notebooks/43337/ ;
Fall 2025 advice https://www.metaculus.com/notebooks/43357/ ; Spring 2026 analysis https://www.metaculus.com/notebooks/45373/ ;
Spring 2026 survey https://www.metaculus.com/notebooks/45382/ ; Spring 2026 advice https://www.metaculus.com/notebooks/45336/ ;
search providers https://www.metaculus.com/notebooks/43361/ ; calibration https://www.metaculus.com/notebooks/43356/ ;
synthesis https://www.lesswrong.com/posts/a82q6yd8zKpYk56cF/ .

| Evidence | Design choice |
|---|---|
| "The most important factor ... is the underlying model"; frontier model is "table stakes"; high-reasoning variants beat standard ones (p = 0.004). | Pin current flagship models with high reasoning effort; never use the library's gpt-4o default. |
| "running multiple forecasts (5+) and taking a median/aggregate was broadly endorsed"; "Cutting forecast counts to save cost backfired badly"; 86% of Fall 2025 winners ensemble; "3-7 diverse runs across model families". | 5 forecasts per question across three labs (OpenAI, Anthropic, Google); median for binary. |
| Research breadth is the strongest survey result: winners used 1.75 sources vs 1.00 (r = 0.42, p = 0.006); no single provider wins; with DeepSeek-R1 in Fall 2025 a native search model ranked 26th, AskNews 58th, no research 65th (negative). | Two native-search sources from different engines, plus AskNews if the owner gets access. Sources fail independently. No research at all = do not forecast (retry, then skip). |
| "capping predictions at a max/min is the strongest within-winners differentiator (r = +0.48, p = 0.005)"; "The community continues to underrate the benefit of clipping"; extremizing correlates negatively (r = -0.30); bots abandoned extremization. | Clip binary to [0.03, 0.97], floor multiple-choice options at 0.02, no extremizing. |
| "Bots mistake unresolved as resolved, producing catastrophic 99% forecasts - add explicit guards". | Prompt says the question is still open; the clip bounds the damage. |
| Parser failures were one maker's "biggest source of errors". | Deterministic regex parser first; the library's LLM parser only as fallback. |
| "Having reliable code that forecasts on every question is very important"; the coverage table above. | Lock, ledger, retries on later timer ticks, per-source and per-model failover. |
| "per-call cost ceiling is not optional"; one maker burned the daily budget in four questions. | Spend meter with per-question, per-run, per-day and season caps; own price table because litellm may not know new models. |
| "Simplicity wins: the template bot is better than I expected. I tried low-effort adjustments but made it worse"; a bot regressed from 20th to 63rd after adding logit aggregation and extremization. | Keep the template's prompts and flow; change only the items above. |
| Numeric: "set wide 90/10 confidence intervals"; numeric distributions tend to be too narrow; Metaculus itself aggregates continuous forecasts by averaging CDFs (code: `utils/the_math/aggregations.py`). | Numeric aggregation = average of the runs' CDFs (a mixture). Under a log score the mixture can never score worse than the average run, and it keeps disagreement between models as width. This is our own reasoning, not a tournament-tested result; the template uses the median CDF. |

## 5. Submission format constraints (code: `questions/serializers/common.py`)

- Binary: `probability_yes` in [0.001, 0.999].
- Multiple choice: every current option present, each in [0.001, 0.999], sum = 1.
- Numeric/discrete `continuous_cdf`: exactly `inbound_outcome_count + 1` values (201 by default); every step at least
  `0.01 / inbound_outcome_count` and at most `0.2 * 200 / inbound_outcome_count`; open lower bound needs
  `cdf[0] >= 0.001`, closed needs `cdf[0] == 0`; open upper bound needs `cdf[-1] <= 0.999`, closed needs `cdf[-1] == 1`.
  `bot/numeric.py::validate_cdf` mirrors this check and is applied before every submission.
