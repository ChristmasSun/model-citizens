# Model Citizens: 2026 Senate forecast

Win probabilities for all 35 Senate races in 2026: 33 regular seats plus the Ohio and Florida specials. Output is `predictions_senate.csv`.

```sh
uv run python fetch.py   # download raw data into data/raw/
uv run python model.py   # fit, simulate, write predictions_senate.csv
```

## Data

| Input | Source | Used for |
|---|---|---|
| Senate and presidential results, 1976–2024 | MIT Election Data + Science Lab | Training targets, state partisan lean, incumbency |
| Final-weeks Senate and generic-ballot polls, with results (1998–2022) | FiveThirtyEight `raw_polls.csv` | Fitting the poll/fundamentals blend and the size of polling errors |
| 2026 Senate polls, generic ballot, Trump approval | VoteHub API (CC BY 4.0) | The 2026 inputs |
| Nominees, incumbency, candidate quality | `data/races_2026.csv` (hand-built from Wikipedia) | Who is running and how strong they are |

Kalshi is **not** used anywhere. It is only the benchmark.

## Method

All margins are measured as (opponent − Republican).

1. **Fundamentals regression.** Fit on every D-vs-R Senate race from 1998–2022 (n ≈ 400), with errors clustered by year.
   - **State lean**: the state's presidential margin minus the national margin, from the last presidential election. A `lean × time` term lets partisanship matter more each cycle. The fitted weight rises from 0.69 in 2010 to about 0.9 by 2022.
   - **Incumbency**, also with a time trend. Its value fell from about 12 points in 2010 to about 1 point by 2022.
   - **National environment**: generic-ballot polling average, presidential approval, and a midterm penalty against the president's party. With the generic ballot in the model, approval and the midterm term add little, because voters' views of the president already show up in the generic ballot.
   - Time trends are evaluated at 2022, the last training cycle, and not extrapolated further.
2. **Candidate quality.** Each nominee is scored 0–4 using the rubric below. The difference between the two candidates is worth `QUALITY_PTS` (1.5) points per tier, scaled down by **polarization**: the full effect applies in a swing state, half applies at 30 points of lean. A great Democrat in Wyoming has fewer persuadable voters to win over than one in Maine.
3. **Incumbent drag (`INCUMBENT_DRAG = 0.5`).** An incumbent from the president's party loses 0.5 points of margin for every point the president's approval is below 50%. With Trump at about 36%, that is about −7 points for every Republican incumbent, which more than cancels their remaining incumbency edge. This is our view that incumbency now ties a candidate to an unpopular president. Across 1998–2022 the data shows no such effect. Fit on 2010–2022 alone, it estimates 0.54 (p = 0.13). If the effect is new, history can't confirm it yet, so we set the value rather than fit it.
4. **Polls.** VoteHub polls from the last 90 days are weighted by recency (14-day half-life), √(sample size), and half weight for partisan or internal polls. Hypothetical matchups and mislabeled polls are dropped by requiring both nominees' names.
5. **Blend.** forecast = w · polls + (1 − w) · fundamentals, where w = n_eff / (n_eff + k). k is fit on 376 polled races from 1998–2022.

   | Approach | RMSE in backtest |
   |---|---|
   | Fundamentals only | 12.8 pts |
   | Polls only | 6.5 pts |
   | Blend | 6.0 pts |

6. **Simulation.** 50,000 simulated elections, each with a **shared national error** (SD 3.0 pts, centered on the historical average miss) plus race-level error (SD 5.3 pts in polled races, 9.0 in unpolled races; both fit on history). Races miss together, as they did in 2016 and 2022.
7. **Wave prior (`WAVE = 2`).** This is our explicit analyst view that the environment is 2 points bluer than the data alone says. It moves the center of the national error. Set it to 0 for a pure-data forecast.

### Candidate quality rubric (0–4)

| Score | Meaning |
|---|---|
| 4 | Former senator or governor (not the incumbent) |
| 3 | Sitting senator, or other statewide office (AG, Lt Gov, at-large House member) |
| 2 | US House member or state legislative leader |
| 1 | State legislator, local official, or prominent first-time candidate |
| 0 | No office and low profile |

Adjustments of ±1 are recorded in the `note` column, for example a record fundraiser (Hamilton, KS) or a major scandal (Paxton, TX).

### Special cases

- **Independents.** NE (Osborn), ID (Achilles), and SD (Bengs) are modeled as the anti-Republican side.
- **Montana.** The four-way race uses whichever of Bodnar (I) or Bankhead (D) leads the polling average. The fundamentals get a −15 point vote-split penalty (`fund_adj`).
- **Florida, Ohio, South Carolina.** The appointed incumbents get no incumbency bonus, because the training data only credits elected incumbents.
