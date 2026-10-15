import pandas as pd
import requests
import time


def fetch_calendar(start_year: int = 2001, end_year: int = 2026) -> pd.DataFrame:
    """Fetch F1 race calendars for a range of seasons from Ergast API.
    Returns:
        pd.DataFrame containing columns [year, round, raceName, circuitId, date].
    """
    data = []
    for year in range(start_year, end_year+1):
        url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
        try:
            r = requests.get(url, timeout=10)
            races = r.json()["MRData"]["RaceTable"]["Races"]
            for race in races:
                data.append({
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "raceName": race["raceName"],
                    "circuitId": race["Circuit"]["circuitId"],
                    "date": race["date"]
                })
        except Exception:
            pass

        time.sleep(0.5)

    return pd.DataFrame(data)
