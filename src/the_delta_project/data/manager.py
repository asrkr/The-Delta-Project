import os
import time
import numpy as np
import pandas as pd

from the_delta_project.data.ergast import fetch_race_result, fetch_qualifying_results
from the_delta_project.data.calendar import fetch_calendar
from the_delta_project.data.sprint import fetch_sprint_results
from the_delta_project.utils.keys import ensure_keys

# -------------------------------------------------------------------
# Absolute paths based on project root
# -------------------------------------------------------------------
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dat"))

RACE_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "race_results.csv")
CALENDAR_CSV_PATH = os.path.join(DATA_DIR, "races_calendar.csv")
SPRINT_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "sprint_results.csv")
QUALI_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "latest_qualifying.csv")


# -------------------------------------------------------------------
# UPDATES (INCREMENTAL LOGIC)
# -------------------------------------------------------------------
def update_race_data(start_year: int = 2001, end_year: int = 2026) -> None:
    """Download race results from Ergast/Jolpica and save them incrementally to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_races = []

    for year in range(start_year, end_year+1):
        cpt = 0
        for rnd in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/results.json"
            result = fetch_race_result(url)

            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame) and not result.empty:
                result["year"] = int(year)
                result["round"] = int(rnd)
                all_races.append(result)
                cpt += 1

            time.sleep(0.5)

    if not all_races:
        return

    df_new = pd.concat(all_races, ignore_index=True)

    if os.path.exists(RACE_RESULTS_CSV_PATH):
        df_old = pd.read_csv(RACE_RESULTS_CSV_PATH)
        # Delete rewritten years to avoid duplicates
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values(["year", "round", "grid"])
    df_final.to_csv(RACE_RESULTS_CSV_PATH, index=False)


def update_calendar(start_year: int = 2001, end_year: int = 2026) -> None:
    """Fetch calendar from Ergast API and save to CSV."""
    df = fetch_calendar(start_year, end_year)
    if not df.empty:
        df.to_csv(CALENDAR_CSV_PATH, index=False)


def update_sprint_dat(start_year: int = 2021, end_year: int = 2026) -> None:
    """Fetch sprint results and update CSV incrementally"""
    if start_year < 2021:
        start_year = 2021

    all_sprints = []
    for year in range(start_year, end_year+1):
        for rnd in range(1, 26):
            df_sprint = fetch_sprint_results(year, rnd)
            if df_sprint is not None and not df_sprint.empty:
                all_sprints.append(df_sprint)
            time.sleep(0.5)

    if not all_sprints:
        return

    df_new = pd.concat(all_sprints, ignore_index=True)

    if os.path.exists(SPRINT_RESULTS_CSV_PATH):
        df_old = pd.read_csv(SPRINT_RESULTS_CSV_PATH)
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values(["year", "round", "sprint_pos"])
    df_final.to_csv(SPRINT_RESULTS_CSV_PATH, index=False)


def update_latest_qualifying(year: int, rnd: int) -> bool:
    """Fetch latest qualifying results for a single race"""
    df = fetch_qualifying_results(year, rnd)
    if df is None or df.empty:
        return True
    df.to_csv(QUALI_RESULTS_CSV_PATH, index=False)
    return True


# -------------------------------------------------------------------
# LOAD + UTILS
# -------------------------------------------------------------------
def load_data() -> pd.DataFrame | None:
    """Load race results dataset and merge calendar (circuitId) and sprint features."""
    if not os.path.exists(RACE_RESULTS_CSV_PATH):
        return None

    df = pd.read_csv(RACE_RESULTS_CSV_PATH)
    df = ensure_keys(df)

    # Merge circuitId via calendar
    if os.path.exists(CALENDAR_CSV_PATH):
        cal = pd.read_csv(CALENDAR_CSV_PATH)
        if "circuitId" not in df.columns:
            df = df.merge(cal[["year", "round", "circuitId"]], on=["year", "round"])

    # Merge sprint features
    sprint_cols = ["sprint_pos", "sprint_grid", "sprint_points"]
    if os.path.exists(SPRINT_RESULTS_CSV_PATH):
        try:
            df_sprint = pd.read_csv(SPRINT_RESULTS_CSV_PATH)
            df_sprint = ensure_keys(df_sprint)
            cols_to_merge = ["year", "round", "DriverKey"] + [c for c in sprint_cols if c in df_sprint.columns]
            df = df.merge(df_sprint[cols_to_merge], on=["year", "round", "DriverKey"], how="left")
        except Exception:
            for c in sprint_cols:
                df[c] = np.nan
    else:
        for c in sprint_cols:
            df[c] = np.nan

    return ensure_keys(df)


def get_rounds_for_race(race_name_keyword: str) -> tuple[dict[int, int], str | None]:
    """Return a mapping of {year: round} and the official race name for a keyword"""
    if not os.path.exists(CALENDAR_CSV_PATH):
        update_calendar()

    if not os.path.exists(CALENDAR_CSV_PATH):
        return {}, None

    df = pd.read_csv(CALENDAR_CSV_PATH)
    filtered = df[df["RaceName"].str.contains(race_name_keyword, case=False, na=False)]

    if filtered.empty:
        return {}, None

    rounds_map = dict(zip(filtered["year"], filtered["round"]))
    official_name = filtered.iloc[0]["raceName"]
    return rounds_map, official_name
