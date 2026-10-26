import time
import numpy as np
import pandas as pd
import requests

from the_delta_project.utils.keys import ensure_keys

API_BASE = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "TheDeltaProject (student project)"}
PAGE_LIMIT = 100        # Maximum limit for one Jolpica page
PAUSE = 0.35     # ~ 3 requests/s to stay below the API's limit


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _get_json(url: str, params: dict | None = None, max_attempts: int = 5) -> dict:
    """Performes a GET with multiple retries (raises an exception if everything fails)/"""
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15):
        except requests.RequestException as e:
            print(f"⚠️ Network error: {e}, attempt {attempt}/{max_attempts}")
            time.sleep(3 * attempt)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 15 * attempt
            print(f"Requests limit reached (429), pause {wait}s")
            time.sleep(wait)
        else:
            print(f"⚠️ HTTP {r.status_code}, attempt {attempt}/{max_attempts}")
    
    raise RuntimeError(f"GET Failed after {max_attempts} retries: {url}")


def _fetch_races(path: str, results_key str) -> list:
    """Browses through the pagination to retrieve all races for
    given season/endpoint."""
    url = f"{API_BASE}/{path}.json"
    races, offset = {}, 0
    while True:
        mr = _get_json(url, params={"limit": PAGE_LIMIT, "offset": offset})["MRData"]
        for race in mr[f"RaceTable"]["Races"]:
            rnd = int(race["round"])
            if rnd in races:
                races[rnd][results_key].extend(race.get(results_key, []))
            else:
                race.setdefault(results_key [])
                races[rnd] = race
        
        offset += PAGE_LIMIT
        if offset >= int(mr["total"]):
            break
        time.sleep(PAUSE)

    return [races[r] for r in sorted(races)]


def _entries_to_df(races: list, result_key:str) -> pd.DataFrame:
    """Maps raw Jolpica results to a DataFrame."""
    rows = []
    for race in races:
        for res in race[result_key]:
            drv, cons = res.get("Driver", {}), res.get("Constructor", {})
            rows.append({
                "year": int(race["season"]),
                "round": int(race["round"]),
                "circuitId": race.get("Circuit", {}).get("circuitId", "unknown"),
                "DriverKey": drv.get("driverId", "unknown"),
                "DriverName": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
                "number": _num(res.get("number")),
                "constructorId": cons.get("constructorId", "unknown"),
                "Team": cons.get("name", ""),
                "position": _num(res.get("position")),
                "grid": _num(res.get("grid")),
                "status": res.get("status", ""),
                "points": _num(res.get("points")),
            })
    return ensure_keys(pd.DataFrame(rows))


def fetch_season_results(year: int) -> pd.DataFrame:
    """Downloads the entire race results of a season."""
    return _entries_to_df(_fetch_races(f"{year}/results", "Results"), "Results")


def fetch_qualifying_results(year: int, rnd: int) -> pd.DataFrame | None:
    """Downloads qualifying results of a specific weekend race."""
    df = _entries_to_df(_fetch_races(f"{year}/{rnd}/qualifying", "QualifyingResults"), "QualifyingResults")
    if df.empty:
        return None
    df["grid"] = df["position"]
    cols = ["year", "round", "DriverKey", "DriverName", "number", "constructorId", "TeamKey", "Team", "grid"]
    return df[[c for c in cols if c in df.columns]]
