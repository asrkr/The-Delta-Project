import time
import pandas as pd
import requests
import numpy as np

from the_delta_project.utils.keys import make_driver_key, make_team_key


def fetch_race_result(url: str) -> pd.DataFrame | str | None:
    """Fetch race results for a single round from Ergast/Jolpica API.
    Returns:
        - pd.DataFrame if results exist
        - 'END_OF_SEASON' if no race exists for this round
        - None if fetching failed after retries
    """
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=10)

            if r.status_code == 200:
                data = r.json()
                races = data["MRData"]["RaceTable"]["Races"]
                if not races:
                    return "END_OF_SEASON"

                circuit_id = races[0].get("Circuit", {}).get("circuitId", "unknown")

                df = pd.DataFrame(races[0]["Results"])

                df["DriverKey"] = df["Driver"].apply(
                    lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", ""))
                )
                df["DriverName"] = df["Driver"].apply(
                    lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip()
                )

                df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
                df["TeamKey"] = df["Team"].apply(make_team_key)

                df["circuitId"] = circuit_id
                df["grid"] = pd.to_numeric(df.get("grid", np.nan), errors="coerce")
                df["position"] = pd.to_numeric(df.get("position", np.nan), errors="coerce")

                cols = ["DriverKey", "DriverName", "TeamKey", "Team", "grid", "position", "status", "points", "circuitId"]
                final_cols = [c for c in cols if c in df.columns]
                return df[final_cols]
            
            elif r.status_code == 429:
                time.sleep((attempt + 1) * 5)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return None


def fetch_qualifying_results(year: int, rnd: int) -> pd.DataFrame | None:
    """Fetch real qualifying results (grid) for a given race from Ergast API.
    Returns:
        - pd.DataFrame if qualifying results exist
        - None if fetching failed or no results found
    """
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/qualifying.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        races = r.json()["MRData"]["RaceTable"]["Races"]
        if not races:
            return None

        quali_results = races[0].get("QualifyingResults", [])
        if not quali_results:
            return None

        df = pd.DataFrame(quali_results)

        df["DriverKey"] = df["Driver"].apply(
            lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", ""))
        )
        df["DriverName"] = df["Driver"].apply(
            lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip()
        )

        df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
        df["TeamKey"] = df["Team"].apply(make_team_key)

        df["grid"] = pd.to_numeric(df["position"], errors="coerce")

        df["year"] = int(year)
        df["round"] = int(rnd)

        return df[["DriverKey", "DriverName", "TeamKey", "Team", "grid", "year", "round"]]

    except Exception:
        return None
