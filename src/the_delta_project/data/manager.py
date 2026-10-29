import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

from the_delta_project.data.calendar import fetch_season_calendar
from the_delta_project.data.ergast import (
    PAUSE,
    fetch_qualifying_results,
    fetch_season_results,
)
from the_delta_project.data.sprint import fetch_season_sprints
from the_delta_project.utils.keys import ensure_keys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../data"))

RACE_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "race_results.csv")
CALENDAR_CSV_PATH = os.path.join(DATA_DIR, "races_calendar.csv")
SPRINT_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "sprint_results.csv")
QUALI_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "latest_qualifying.csv")


def _current_year() -> int:
    """Return the current year."""
    return datetime.now().year


def _save_season(path: str, df_new: pd.DataFrame, years: list[int], sort_cols: list[str]) -> None:
    """Save newly fetched season DataFrames, replacing existing rows for the given years."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        if len(old) and "constructorId" in df_new.columns and "constructorId" not in old.columns:
            raise RuntimeError(
                f"{os.path.basename(path)} is in the old format. Move it and re-run full update."
            )
        old = old[~old["year"].isin(years)]
        df_new = pd.concat([old, df_new], ignore_index=True)
    else:
        df_new = df_new
    
    df_new.sort_values(sort_cols).to_csv(path, index=False)


def _update_by_season(label: str, fetch_fn, path: str, start_year: int, end_year: int | None, sort_cols: list[str]) -> bool:
    """Update dataset season by season, replacing complete seasons to avoid duplicates."""
    target_end_year = end_year or _current_year()
    print(f"📌 {label} {start_year}-{end_year}")
    done, failed = [], []

    for year in range(start_year, target_end_year + 1):
        try:
            df_year = fetch_fn(year)
        except Exception as e:
            print(f"❌ {year}: {e} -> kept old data")
            failed.append(year)
            continue

        if df_year is None or df_year.empty:
            print(f"{year}: no data")
            continue

        _save_season(path, df_year, [year], sort_cols)
        print(f"{year}: {df_year['round'].nunique()} races ✔️")
        done.append(year)
        time.sleep(PAUSE)
    
    if failed:
        print(f"⚠️ Failed seasons: {failed}")
    elif done:
        print(f"✔️ Saved -> {path}")
    
    return not failed


def update_database(start_year: int = 2001, end_year: int | None = None) -> bool:
    """Fetch and update full race results from Jolpica API."""
    return _update_by_season(
        "Ergast race results",
        fetch_season_results,
        RACE_RESULTS_CSV_PATH,
        start_year,
        end_year,
        ["year", "round", "position"]
    )


def update_sprint_data(start_year: int = 2021, end_year: int | None = None) -> bool:
    """Fetch and update sprint race results from Jolpica API."""
    return _update_by_season(
        "Ergast sprint results",
        fetch_season_sprints,
        SPRINT_RESULTS_CSV_PATH,
        max(start_year, 2021),
        end_year,
        ["year", "round"],
    )


def update_calendar(start_year: int = 2001, end_year: int | None = None) -> bool:
    """Fetch and update F1 season calendars from Jolpica API."""
    return _update_by_season(
        "Calendar",
        fetch_season_calendar,
        CALENDAR_CSV_PATH,
        start_year,
        end_year,
        ["year", "round"],
    )


def update_latest_qualifying(year: int, rnd: int) -> bool:
    """"Fetch latest qualifying results for a single Grand Prix and save to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        df = fetch_qualifying_results(year, rnd)
    except Exception as e:
        print(f"❌ Qualif {year} R{rnd}: {e}")
        return False
    
    if df is None or df.empty:
        print(f"❌ No qualifying session found for {year} R{rnd}")
        return False
    
    df.to_csv(QUALI_RESULTS_CSV_PATH, index=False)
    print(f"✔️ Qualifying {year} R{rnd} saved.")
    return True


def load_data() -> pd.DataFrame | None:
    """Load race results dataset and merge calendar and sprint features."""
    if not os.path.exists(RACE_RESULTS_CSV_PATH):
        print("File not found.")
        return None
    
    df = pd.read_csv(RACE_RESULTS_CSV_PATH)
    if "constructorId" not in df.columns:
        print(f"❌ {os.path.basename(RACE_RESULTS_CSV_PATH)} is in old format. Run full update.")
        return None
    
    df = ensure_keys(df)

    sprint_cols = ["sprint_pos", "sprint_grid", "sprint_points"]
    if os.path.exists(SPRINT_RESULTS_CSV_PATH):
        try:
            df_sprint = pd.read_csv(SPRINT_RESULTS_CSV_PATH)
            cols_to_merge = ["year", "round", "DriverKey"] + [
                c for c in sprint_cols if c in df_sprint.columns
            ]
            df = df.merge(df_sprint[cols_to_merge], on=["year", "round", "DriverKey"], how="left")
        except Exception as e:
            print(f"⚠️ Error loading sprint data: {e}")
            for c in sprint_cols:
                df[c] = np.nan
    
    else:
        for c in sprint_cols:
            df[c] = np.nan
    
    return df


def get_rounds_for_race(race_name_keyword: str) -> tuple[dict[int, int], str | None]:
    """Return a mapping of {year: round} and official race name matching keyword."""
    if not os.path.exists(CALENDAR_CSV_PATH):
        print("Caledndar not found. Downloading...")
        update_calendar()
    
    if not os.path.exists(CALENDAR_CSV_PATH):
        return {}, None
    
    df = pd.read_csv(CALENDAR_CSV_PATH)
    filtered = df[df["raceName"].str.contains(race_name_keyword, case=False, na=False)]

    if filtered.empty:
        print(f'No race found with name "{race_name_keyword}"')
        return {}, None
    
    rounds_map = dict(zip(filtered["year"], filtered["round"]))
    official_name = filtered.iloc[0]["raceName"]

    return rounds_map, official_name


def get_race_participants(df: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Get participant list for a specific race round."""
    r = df[(df["year"] == year) & (df["round"] == rnd)].sort_values("grid")
    if not r.empty:
        r = ensure_keys(r)
        if "grid" in r.columns:
            r = r.sort_values("grid")
        cols = ["DriverKey", "Team", "TeamKey", "constructorId", "grid"]
        final_cols = [c for c in cols if c in r.columns]
        return r[final_cols].drop_duplicates()
    
    entry_file = os.path.join(DATA_DIR, f"entry_list_{year}.csv")
    if os.path.exists(entry_file):
        entry_df = pd.read_csv(entry_file)
        return ensure_keys(entry_df)
    
    return pd.DataFrame()


def has_real_qualifying(year: int, rnd: int) -> bool:
    """Check if real qualifying results exist for a given race round."""
    if not os.path.exists(QUALI_RESULTS_CSV_PATH):
        return False
    try:
        df_q = pd.read_csv(QUALI_RESULTS_CSV_PATH)
        if "year" not in df_q.columns or "round" not in df_q.columns:
            return False
        mask = (df_q["year"] == year) & (df_q["round"] == rnd)
        return not df_q[mask].empty
    except Exception:
        return False


def load_real_qualifying(year: int, rnd: int) -> pd.DataFrame:
    """Load real qualifying results for a given race round."""
    if not os.path.exists(QUALI_RESULTS_CSV_PATH):
        return pd.DataFrame()
    
    df_q = pd.read_csv(QUALI_RESULTS_CSV_PATH)
    df_q = ensure_keys(df_q)

    mask = (df_q["year"] == year) & (df_q["round"] == rnd)
    quali = df_q[mask].copy()

    cols = [
        c for c in [
            "DriverKey", "DriverName",
            "number", "constructorId",
            "TeamKey", "Team",
            "grid", "year", "round",
        ] if c in quali.columns
    ]
    return quali[cols]
