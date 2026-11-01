"""2026 Senate forecast: fundamentals + polls + correlated simulation.

All margins are (opponent − Republican) in points, so positive = the R loses.
Run `uv run python fetch.py` once, then `uv run python model.py`.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
TODAY = date.today()

# Analyst prior: how many points we believe the national environment is *bluer*
# than polls + fundamentals say. 0 = pure data. This is the "blue wave" dial.
WAVE = 2.0
# Points of margin per tier of candidate-quality difference (see README rubric),
# before polarization damping. No clean historical data, so this is a prior.
QUALITY_PTS = 1.5
# Quality effects shrink in lopsided states (polarization: few persuadable voters left).
# Full effect in a swing state, half at this many points of partisan lean.
POLARIZATION_HALF = 30
POLL_HALF_LIFE_DAYS = 14
# Analyst prior: an incumbent of the president's party loses this many points of margin per point
# the president's approval is below 50%. Our view is that this is a newer phenomenon than the
# training data can confirm (2010-2022 alone estimates ~0.54, p=0.13), so it is set, not fitted.
INCUMBENT_DRAG = 0.5
# Time trends (polarization) are evaluated at the last training cycle (2022), not extrapolated.
T_FORECAST = 1.2
# Seats not up in 2026, and the threshold for control (VP Vance breaks ties for R).
HOLDOVER = {"R": 31, "D": 34}
N_SIMS = 50_000

# Gallup-style presidential approval just before each election (approximate).
APPROVAL = {1998: 66, 2000: 57, 2002: 63, 2004: 48, 2006: 38, 2008: 25, 2010: 45,
            2012: 52, 2014: 42, 2016: 55, 2018: 41, 2020: 46, 2022: 41, 2024: 39}
PRES_PARTY = {y: s for s, ys in {+1: [1998, 2000, 2010, 2012, 2014, 2016, 2022, 2024],
                                 -1: [2002, 2004, 2006, 2008, 2018, 2020]}.items() for y in ys}


# ---------- history ----------

def state_lean() -> pd.DataFrame:
    """Each state's presidential margin minus the national margin (D − R), per election year."""
    p = pd.read_csv(RAW / "president.csv")
    tot = p.groupby(["year", "state_po"]).totalvotes.first()
    two = p[p.party_simplified.isin(["DEMOCRAT", "REPUBLICAN"])].pivot_table(
        index=["year", "state_po"], columns="party_simplified", values="candidatevotes", aggfunc="sum")
    margin = (two.DEMOCRAT - two.REPUBLICAN) / tot * 100
    natl = (two.DEMOCRAT - two.REPUBLICAN).groupby("year").sum() / tot.groupby("year").sum() * 100
    return (margin - natl.reindex(margin.index.get_level_values("year")).values).rename("lean").reset_index()


def lean_before(leans: pd.DataFrame, year: int) -> pd.Series:
    """Lean from the most recent presidential election strictly before `year` (what a forecaster knows)."""
    py = leans.year[leans.year < year].max()
    return leans[leans.year == py].set_index("state_po").lean


def national_env() -> pd.DataFrame:
    """Per cycle: final-weeks generic-ballot polling average, plus approval and midterm signals."""
    r = pd.read_csv(RAW / "raw_polls.csv")
    gb = r[r.type_simple == "House-G-US"].groupby("cycle").margin_poll.mean().rename("gb")
    env = gb.to_frame()
    env["pres"] = env.index.map(PRES_PARTY)  # +1 if D president
    # Signed so positive = good for Democrats.
    env["approval"] = env.pres * (env.index.map(APPROVAL) - 50)
    env["midterm"] = -env.pres * (env.index % 4 == 2)
    return env


def senate_history(leans: pd.DataFrame) -> pd.DataFrame:
    """One row per D-vs-R Senate general election (1998-2022) with fundamentals features."""
    s = pd.read_csv(RAW / "senate.csv")
    s = s[s.stage.str.lower() == "gen"]
    s = s.groupby(["year", "state_po", "special", "candidate", "party_simplified"], as_index=False).agg(
        votes=("candidatevotes", "sum"), total=("totalvotes", "first"))
    rows = []
    for (y, st, sp), g in s.groupby(["year", "state_po", "special"]):
        top = g.nlargest(2, "votes")
        d, r = top[top.party_simplified == "DEMOCRAT"], top[top.party_simplified == "REPUBLICAN"]
        if len(d) != 1 or len(r) != 1:
            continue  # uncontested, or a top-two that isn't D vs R
        winner = top.iloc[0]
        rows.append(dict(year=y, state=st, special=sp, d=d.candidate.iloc[0], r=r.candidate.iloc[0],
                         winner=winner.candidate, margin=(d.votes.iloc[0] - r.votes.iloc[0]) / winner.total * 100))
    h = pd.DataFrame(rows)

    # Elected incumbent = won a Senate race in this state within the previous six years.
    wins = h.groupby(["state", "winner"]).year.apply(list).to_dict()
    h["inc"] = [next((sign for name, sign in ((row.d, 1), (row.r, -1))
                      if any(row.year - 6 <= y < row.year for y in wins.get((row.state, name), []))), 0)
                for row in h.itertuples()]
    h["lean"] = [lean_before(leans, row.year).get(row.state, np.nan) for row in h.itertuples()]
    h["t"] = (h.year - 2010) / 10  # decades from 2010, for polarization trends
    env = national_env()
    h = h.join(env, on="year", how="inner")
    return h.dropna(subset=["lean"])


def fit_fundamentals(h: pd.DataFrame):
    """
    margin ~ lean (stronger over time = polarization) + incumbency (weaker over time and in
    lopsided states) + generic ballot + approval + midterm penalty.
    """
    h = h.assign(abslean=h.lean.abs())
    return smf.ols("margin ~ lean + lean:t + inc + inc:t + inc:abslean + gb + approval + midterm", h).fit(
        cov_type="cluster", cov_kwds={"groups": h.year})


# ---------- polls ----------

def poll_weight(days_old: np.ndarray, n: np.ndarray, partisan: np.ndarray) -> np.ndarray:
    """Recency half-life × sqrt(sample size) × half weight for partisan/internal polls."""
    return 0.5 ** (days_old / POLL_HALF_LIFE_DAYS) * np.sqrt(np.clip(n, 100, 3000)) * np.where(partisan, 0.5, 1.0)


def historical_poll_avgs() -> pd.DataFrame:
    """538 final-weeks Senate polls → one weighted average (D − R) per race, with the actual result."""
    r = pd.read_csv(RAW / "raw_polls.csv")
    r = r[(r.type_simple == "Sen-G") & (r.cand1_party == "DEM") & (r.cand2_party == "REP")]
    r = r.assign(w=poll_weight(r.time_to_election.to_numpy(), r.samplesize.fillna(500).to_numpy(),
                               r.partisan.notna().to_numpy()))
    return r.groupby(["cycle", "location", "race"]).apply(
        lambda g: pd.Series(dict(poll=np.average(g.margin_poll, weights=g.w), n_eff=g.w.sum() / g.w.max(),
                                 actual=g.margin_actual.iloc[0])), include_groups=False).reset_index()


def blend_weight(n_eff, k: float):
    """Share of the forecast that comes from polls rather than fundamentals."""
    return n_eff / (n_eff + k)


def fit_blend(h: pd.DataFrame, fund) -> dict:
    """Pick the poll/fundamentals blend and error sizes from 1998-2022 polled races."""
    ph = historical_poll_avgs()
    h = h.assign(fund=fund.predict(h.assign(abslean=h.lean.abs())))
    # Join each polled race to its fundamentals row: same year/state, closest actual margin.
    m = ph.merge(h, left_on=["cycle", "location"], right_on=["year", "state"])
    m = m.loc[(m.actual - m.margin).abs().groupby([m.cycle, m.race]).idxmin()]
    m = m[(m.actual - m.margin).abs() < 1]

    def rmse(k):
        w = blend_weight(m.n_eff, k)
        return np.sqrt(np.mean((w * m.poll + (1 - w) * m.fund - m.actual) ** 2))

    k = min(np.arange(0.25, 20, 0.25), key=rmse)
    w = blend_weight(m.n_eff, k)
    resid = w * m.poll + (1 - w) * m.fund - m.actual
    natl = resid.groupby(m.cycle).mean()  # errors every race in a cycle shared
    recent = h.year >= 2014
    fund_resid = (h.margin - h.fund)[recent]
    return dict(
        k=k, bias_natl=natl.mean(), n_races=len(m), rmse_blend=rmse(k), rmse_polls=np.sqrt(np.mean((m.poll - m.actual) ** 2)),
        rmse_fund=np.sqrt(np.mean((m.fund - m.actual) ** 2)),
        sd_natl=natl.std(),
        sd_race_polled=(resid - natl.reindex(m.cycle).values).std(),
        sd_race_unpolled=(fund_resid - fund_resid.groupby(h.year[recent]).transform("mean")).std(),
    )


# ---------- 2026 ----------

def votehub() -> pd.DataFrame:
    v = pd.read_json(RAW / "votehub.json")
    v["end_date"] = pd.to_datetime(v.end_date).dt.date
    v["days_old"] = [(TODAY - d).days for d in v.end_date]
    return v


def topline(v: pd.DataFrame, poll_type: str, subject: str, score) -> float:
    """Weighted average over the last 45 days of `score(answers)` for national polls."""
    v = v[(v.poll_type == poll_type) & (v.subject == subject) & (v.days_old <= 45)]
    vals = v.answers.map(score)
    w = poll_weight(v.days_old.to_numpy(), v.sample_size.fillna(1000).to_numpy(),
                    (v.partisan.notna() | v.internal).to_numpy())
    return float(np.average(vals, weights=w))


def pct(answers, name: str) -> float:
    return next((a["pct"] for a in answers if a["choice"].lower().startswith(name.lower())), np.nan)


def race_polls(v: pd.DataFrame, race: pd.Series) -> tuple[float, float, str]:
    """Weighted (opponent − R) polling margin, effective poll count, and which opponent leads."""
    state = race.state_name
    sv = v[(v.poll_type == "us-senator") & (v.subject == f"2026 {state}") & (v.days_old <= 90)]
    opps = race.opp_name.split("|")
    rows = []
    for p in sv.itertuples():
        r = pct(p.answers, race.r_name)
        o = {name: pct(p.answers, name) for name in opps}
        if np.isnan(r) or all(np.isnan(x) for x in o.values()):
            continue  # hypothetical matchup or a mislabeled poll
        lead = max((x for x in o.items() if not np.isnan(x[1])), key=lambda x: x[1])
        n = 600 if pd.isna(p.sample_size) else p.sample_size
        rows.append((lead[1] - r, lead[0], p.days_old, n, pd.notna(p.partisan) or bool(p.internal)))
    if not rows:
        return np.nan, 0.0, opps[0]
    m, lead, d, n, part = map(np.array, zip(*rows))
    w = poll_weight(d.astype(float), n.astype(float), part.astype(bool))
    leader = pd.Series(w).groupby(lead).sum().idxmax()
    return float(np.average(m, weights=w)), float(w.sum() / w.max()), leader


STATES = {"AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "CO": "Colorado", "DE": "Delaware",
          "FL": "Florida", "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IA": "Iowa",
          "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MA": "Massachusetts",
          "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MT": "Montana", "NE": "Nebraska",
          "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NC": "North Carolina",
          "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "RI": "Rhode Island", "SC": "South Carolina",
          "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "VA": "Virginia",
          "WV": "West Virginia", "WY": "Wyoming"}


def forecast(fund, blend: dict) -> pd.DataFrame:
    v = votehub()
    races = pd.read_csv(ROOT / "data" / "races_2026.csv", keep_default_na=False)
    races["state"] = races.race_id.str[:2]
    races["state_name"] = races.state.map(STATES)

    gb = topline(v, "generic-ballot", "2026", lambda a: pct(a, "Dem") - pct(a, "Rep"))
    approve = topline(v, "approval", "Donald Trump", lambda a: pct(a, "Approve"))
    env = dict(gb=gb, approval=-(approve - 50), midterm=1.0, t=T_FORECAST)  # R president, midterm

    lean = lean_before(state_lean(), 2026)
    races["lean"] = races.state.map(lean)
    races["inc"] = races.incumbent.map({"R": -1, "D": 1, "I": 1}).fillna(0)
    races["abslean"] = races.lean.abs()
    races["damp"] = 1 / (1 + races.abslean / POLARIZATION_HALF)
    base = fund.predict(races.assign(**env))
    races["quality"] = QUALITY_PTS * (races.opp_quality - races.r_quality) * races.damp
    # R incumbents share Trump's unpopularity (inc = -1, approval < 50 → shifts margin toward the opponent).
    races["drag"] = INCUMBENT_DRAG * races.inc * (races.inc == -1) * (approve - 50)
    races["fund"] = base + races.quality + races.drag + races.fund_adj.astype(float)

    polls = [race_polls(v, r) for _, r in races.iterrows()]
    races["poll"], races["n_eff"], races["opp_lead"] = zip(*polls)
    w = blend_weight(races.n_eff, blend["k"]).where(races.poll.notna(), 0.0)
    races["w_polls"] = w
    races["mean"] = w * races.poll.fillna(0) + (1 - w) * races.fund
    races["sd_race"] = np.where(w > 0, blend["sd_race_polled"], blend["sd_race_unpolled"])

    rng = np.random.default_rng(2026)
    # Shared national error: centered on the historical miss (polls+fundamentals vs. results) plus our wave prior.
    natl = rng.normal(-blend["bias_natl"] + WAVE, blend["sd_natl"], N_SIMS)[:, None]
    sims = races["mean"].to_numpy() + natl + rng.normal(0, 1, (N_SIMS, len(races))) * races.sd_race.to_numpy()
    races["p_opp"] = (sims > 0).mean(axis=0)
    races["opp_party_lead"] = [p.split("|")[n.split("|").index(l)] for p, n, l in
                               zip(races.opp_party, races.opp_name, races.opp_lead)]
    non_r_seats = HOLDOVER["D"] + (sims > 0).sum(axis=1)
    races.attrs.update(gb=gb, approve=approve, p_control=(non_r_seats >= 51).mean(),
                       median_seats=int(np.median(non_r_seats)))
    return races


def main():
    leans = state_lean()
    h = senate_history(leans)
    fund = fit_fundamentals(h)
    blend = fit_blend(h, fund)
    print(fund.summary().tables[1])
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in blend.items()})

    races = forecast(fund, blend)
    print(f"\n2026 environment: generic ballot D{races.attrs['gb']:+.1f}, Trump approval {races.attrs['approve']:.1f}%, wave prior D+{WAVE}")

    out = pd.DataFrame({
        "race_id": races.race_id,
        "chamber": "Senate",
        "predicted_winner_party": np.where(races.p_opp > 0.5, races.opp_party_lead, "R"),
        "candidate_name": np.where(races.p_opp > 0.5, races.opp_lead, races.r_name),
        "win_probability": np.round(np.maximum(races.p_opp, 1 - races.p_opp), 3),
    })
    out.to_csv(ROOT / "predictions_senate.csv", index=False)

    show = races[["race_id", "lean", "drag", "fund", "poll", "w_polls", "mean", "p_opp"]].copy()
    show["pick"] = out.predicted_winner_party.values
    print(show.sort_values("p_opp").round(2).to_string(index=False))
    print(f"\nD/I seats (median): {races.attrs['median_seats']}  P(D/I control, 51+): {races.attrs['p_control']:.2f}")


if __name__ == "__main__":
    main()
