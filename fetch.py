"""Download every raw input into data/raw/. Run once before model.py."""

import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"

SOURCES = {
    # MIT Election Data + Science Lab, 1976-2024
    "president.csv": "https://dataverse.harvard.edu/api/access/datafile/13887042",
    "senate.csv": "https://dataverse.harvard.edu/api/access/datafile/13887039?format=original",
    # FiveThirtyEight: every poll in the final weeks of a race, joined to the actual result (1998-2022)
    "raw_polls.csv": "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv",
    # VoteHub (CC BY 4.0): 2026 Senate polls, Trump approval, generic ballot
    "votehub.json": "https://api.votehub.com/polls",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "model-citizens/0.1"})
        with urllib.request.urlopen(req) as r:
            (RAW / name).write_bytes(r.read())
        print(f"fetched {name}")
