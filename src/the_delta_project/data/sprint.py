import time
import pandas as pd
import requests
import numpy as np

from the_delta_project.utils.keys import make_driver_key, make_team_key


def fetch_sprint_results(year: int, rnd: int) -> pd.DataFrame | None:
    if year < 2021:
        return None

    url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/sprint.json"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        races = r.json()["MRData"]["RaceTable"]["Races"]
        if not races:
            return None

        sprint_results = races[0].get("SprintResults", [])
        if not sprint_results:
            return None

        circuit_id = races[0].get("Circuit", {}).get("circuitId", "unknown")
        df = pd.DataFrame(sprint_results)

        df["DriverKey"] = df["Driver"].apply(
            lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", ""))
        )
        df["DriverName"] = df["Driver"].apply(
            lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip()
        )

        df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
        df["TeamKey"] = df["Team"].apply(make_team_key)

        df["sprint_pos"] = pd.to_numeric(df.get("position", np.nan), errors="coerce")
        df["sprint_grid"] = pd.to_numeric(df.get("grid", np.nan), errors="coerce")
        df["sprint_points"] = pd.to_numeric(df["points"], errors="coerce")

        df["year"] = int(year)
        df["round"] = int(rnd)
        df["circuitId"] = circuit_id

        cols = ["DriverKey", "DriverName", "TeamKey", "Team", "sprint_grid", "sprint_pos", "status", "sprint_points", "year", "round"]
        final_cols = [c for c in cols if c in df.columns]
        return df[final_cols]
        
    except Exception:
        return None
