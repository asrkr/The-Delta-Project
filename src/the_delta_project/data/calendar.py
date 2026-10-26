import pandas as pd

from the_delta_project.data.ergast import PAGE_LIMIT, API_BASE, _get_json


def fetch_season_calendar(year: int) -> pd.DataFrame:
    """Fetch official race calendar for a single F1 season from Jolpica API."""
    url = f"{API_BASE}/{year}.json"
    mr = _get_json(url, params={"limit": PAGE_LIMIT})["MRData"]

    races = mr.get("RaceTable", {}).get("Races", [])
    data = [
        {
            "year": int(r["season"]),
            "round": int(r["round"]),
            "raceName": r["raceName"],
            "circuitId": r["Circuit"]["circuitId"],
            "date": r["date"],
        }
        for r in races
    ]

    return pd.DataFrame(data)
