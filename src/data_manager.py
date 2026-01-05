import pandas as pd
import requests
import time
import os
import fastf1
import numpy as np
import unicodedata
import re

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")

RESULTS_CSV_PATH = os.path.join(DATA_DIR, "f1_data_complete.csv")
CALENDAR_CSV_PATH = os.path.join(DATA_DIR, "races_calendar.csv")
EXTRA_CSV_PATH = os.path.join(DATA_DIR, "f1_extra_features.csv")
QUALI_CSV_PATH = os.path.join(DATA_DIR, "latest_qualifying.csv")
SPRINT_CSV_PATH = os.path.join(DATA_DIR, "f1_sprint_results.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

# -------------------------------------------------------------------
# DRIVER KEY CREATION
# -------------------------------------------------------------------

# Driver key aliases (cross-source canonicalization)
DRIVER_KEY_ALIASES = {
    "a_antonelli": "k_antonelli",
}

def canonicalize_driver_key(raw_key: str) -> str:
    if raw_key is None:
        return "unknown"
    s = str(raw_key).strip().lower()
    return DRIVER_KEY_ALIASES.get(s, s)

def make_driver_key(given_name: str, family_name: str) -> str:
    """
    Build a stable driver key: first initial + '_' + familyname (ascii, lowercase).
    Then canonicalize via DRIVER_KEY_ALIASES (to fix cross-source mismatches).
    """
    if not given_name or not family_name:
        return "unknown"

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-zA-Z]", "", s)
        return s.lower()

    g = normalize(given_name)
    f = normalize(family_name)
    if not g or not f:
        return "unknown"

    driver_key = f"{g[0]}_{f}"
    return canonicalize_driver_key(driver_key)

# -------------------------------------------------------------------
# TEAM KEY CREATION
# -------------------------------------------------------------------

TEAM_KEY_ALIASES = {
    # Benetton -> Renault -> (Lotus F1) -> Renault -> Alpine
    "benetton": "alpine",
    "renault": "alpine",
    "alpine": "alpine",
    "alpine_f1_team": "alpine",
    "lotus_f1": "alpine",
    "lotus_f1_team": "alpine",

    # Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
    "jordan": "aston_martin",
    "midland": "aston_martin",
    "spyker": "aston_martin",
    "spyker_mf1": "aston_martin",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",

    # Sauber family (BMW Sauber / Alfa Romeo / Kick Sauber / Audi)
    "sauber": "sauber",
    "bmw_sauber": "sauber",
    "kick_sauber": "sauber",
    "alfa_romeo": "sauber",
    "alfa_romeo_racing": "sauber",
    "audi": "sauber",

    # Red Bull family
    "jaguar": "red_bull",
    "red_bull": "red_bull",
    "red_bull_racing": "red_bull",

    # RB junior (Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls)
    "minardi": "rb_junior",
    "toro_rosso": "rb_junior",
    "alphatauri": "rb_junior",
    "rb": "rb_junior",
    "rb_f1_team": "rb_junior",
    "racing_bulls": "rb_junior",

    # Misc normalization for modern naming variants
    "haas_f1_team": "haas",
    "haas": "haas",
}

def make_team_key(team_name: str) -> str:
    """
    Returns a stable TeamKey from a display name (Team).
    Team stays "pretty" for output; TeamKey is used for encoding / history continuity.
    """
    if team_name is None or (isinstance(team_name, float) and pd.isna(team_name)):
        return "unknown"
    s = str(team_name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return TEAM_KEY_ALIASES.get(s, s)

def ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill DriverKey/TeamKey on any dataframe (useful for older CSVs without these cols).
    """
    out = df.copy()

    if "DriverKey" in out.columns:
        out["DriverKey"] = out["DriverKey"].astype(str).apply(canonicalize_driver_key)

    if "TeamKey" not in out.columns and "Team" in out.columns:
        out["TeamKey"] = out["Team"].apply(make_team_key)
    elif "TeamKey" in out.columns:
        out["TeamKey"] = out["TeamKey"].astype(str).apply(make_team_key)

    return out

# -------------------------------------------------------------------
# ERGAST — FETCH
# -------------------------------------------------------------------

def _fetch_race_result(url):
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

                # Safeguard with .get to prevent errors on older seasons
                if "Driver" in df.columns:
                    df["DriverKey"] = df["Driver"].apply(
                        lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", ""))
                    )
                    df["DriverName"] = df["Driver"].apply(
                        lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip()
                    )

                if "Constructor" in df.columns:
                    df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
                    df["TeamKey"] = df["Team"].apply(make_team_key)

                df["circuitId"] = circuit_id
                df["grid"] = pd.to_numeric(df.get("grid", np.nan), errors="coerce")
                df["position"] = pd.to_numeric(df.get("position", np.nan), errors="coerce")

                cols_ok = ["DriverKey", "DriverName", "TeamKey", "Team", "grid", "position", "status", "points", "circuitId"]
                final_cols = [c for c in cols_ok if c in df.columns]
                return df[final_cols]

            elif r.status_code == 429:
                time.sleep((attempt + 1) * 5)
            else:
                time.sleep(2)
        except:
            time.sleep(2)
    return None

def fetch_qualifying_results(year, rnd):
    """
    Retrieves real qualifying results (grid) from the Ergast API.
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

        # IMPORTANT: Team comes from Constructor, then TeamKey computed from Team
        df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
        df["TeamKey"] = df["Team"].apply(make_team_key)

        # Standardize column name for the rest of the pipeline
        df["grid"] = pd.to_numeric(df["position"], errors="coerce")

        # Add year and round to enable filtering later
        df["year"] = int(year)
        df["round"] = int(rnd)

        return df[["DriverKey", "DriverName", "TeamKey", "Team", "grid", "year", "round"]]

    except Exception as e:
        print(f"Error qualifying fetch: {e}.")
        return None

def load_real_qualifying(year, rnd):
    if not os.path.exists(QUALI_CSV_PATH):
        return pd.DataFrame()

    df_q = pd.read_csv(QUALI_CSV_PATH)
    df_q = ensure_keys(df_q)

    mask = (df_q["year"] == year) & (df_q["round"] == rnd)
    quali = df_q[mask].copy()

    cols = [c for c in ["DriverKey", "DriverName", "TeamKey", "Team", "grid", "year", "round"] if c in quali.columns]
    return quali[cols]

def fetch_sprint_results(year, rnd):
    """
    Retrieves Sprint results from Ergast API.
    Returns a DataFrame or None if no sprint occurred during the weekend
    """
    if year < 2021:
        return None

    url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/sprint.json"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        races = data["MRData"]["RaceTable"]["Races"]

        if not races:
            return None

        circuit_id = races[0]["Circuit"]["circuitId"]
        sprint_results = races[0].get("SprintResults", [])
        if not sprint_results:
            return None

        df = pd.DataFrame(sprint_results)

        df["DriverKey"] = df["Driver"].apply(lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", "")))
        df["DriverName"] = df["Driver"].apply(lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip())
        df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
        df["TeamKey"] = df["Team"].apply(make_team_key)

        df["sprint_pos"] = pd.to_numeric(df["position"], errors="coerce")
        df["sprint_grid"] = pd.to_numeric(df["grid"], errors="coerce")
        df["sprint_points"] = pd.to_numeric(df["points"], errors="coerce")

        df["year"] = int(year)
        df["round"] = int(rnd)
        df["circuitId"] = circuit_id

        cols = ["DriverKey", "DriverName", "TeamKey", "Team", "sprint_grid", "sprint_pos", "status", "sprint_points", "year", "round"]
        final_cols = [c for c in cols if c in df.columns]
        return df[final_cols]

    except Exception as e:
        print(f"Error sprint fetch: {e}.")
        return None

# -------------------------------------------------------------------
# ERGAST — UPDATES (WITH INCREMENTAL LOGIC)
# -------------------------------------------------------------------

def update_database(start_year=2001, end_year=2025):
    print(f"📌 Updating Ergast results {start_year}-{end_year}.")
    all_races = []

    for year in range(start_year, end_year + 1):
        print(f" Season {year}...", end=" ")
        cpt = 0
        for rnd in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/results.json"
            result = _fetch_race_result(url)

            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame):
                result["year"] = int(year)
                result["round"] = int(rnd)
                all_races.append(result)
                cpt += 1

            time.sleep(0.7)
        print(f"({cpt} races)")

    if not all_races:
        print("❌ No data downloaded.")
        return

    df_new = pd.concat(all_races, ignore_index=True)

    if os.path.exists(RESULTS_CSV_PATH):
        df_old = pd.read_csv(RESULTS_CSV_PATH)
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    if "grid" in df_final.columns:
        df_final = df_final.sort_values(["year", "round", "grid"])
    else:
        df_final = df_final.sort_values(["year", "round"])

    df_final.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"✔️ Saved → {RESULTS_CSV_PATH}.")

def update_latest_qualifying(year, rnd):
    df = fetch_qualifying_results(year, rnd)
    if df is None:
        return False
    df.to_csv(QUALI_CSV_PATH, index=False)
    return True

def update_sprint_data(start_year=2021, end_year=2025):
    print(f"📌 Updating Sprint Results {start_year}-{end_year}")

    if start_year < 2021:
        start_year = 2021

    all_sprints = []

    for year in range(start_year, end_year+1):
        print(f" Season {year}...", end=" ")
        count = 0
        for rnd in range(1, 26):
            df_sprint = fetch_sprint_results(year, rnd)
            if df_sprint is not None and not df_sprint.empty:
                all_sprints.append(df_sprint)
                count += 1
            time.sleep(0.5)
        print(f"({count} sprints found)")

    if not all_sprints:
        print("No sprint data found.")
        return

    df_new = pd.concat(all_sprints, ignore_index=True)

    if os.path.exists(SPRINT_CSV_PATH):
        df_old = pd.read_csv(SPRINT_CSV_PATH)
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values(["year", "round", "sprint_pos"])
    df_final.to_csv(SPRINT_CSV_PATH, index=False)
    print(f"✔️ Sprint Data saved → {SPRINT_CSV_PATH}")

# -------------------------------------------------------------------
# CALENDAR
# -------------------------------------------------------------------

def update_calendar(start_year=2001, end_year=2025):
    print(f"📌 Updating calendar {start_year}-{end_year}.")
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
        except:
            pass
        time.sleep(0.5)

    pd.DataFrame(data).to_csv(CALENDAR_CSV_PATH, index=False)
    print("✔️ Calendar updated.")

# -------------------------------------------------------------------
# FASTF1 — incrementally append to EXTRA CSV
# -------------------------------------------------------------------

def extract_fastf1_features(start_year, end_year):
    print(f"📌 FastF1 extraction (telemetry) {start_year}-{end_year}.")
    all_entries = []

    if not os.path.exists(CALENDAR_CSV_PATH):
        print("Calendar missing → creating...")
        update_calendar(start_year, end_year)

    calendar = pd.read_csv(CALENDAR_CSV_PATH)

    for year in range(start_year, end_year + 1):
        season = calendar[calendar["year"] == year]
        if season.empty:
            continue

        print(f" Season {year}")
        for _, race in season.iterrows():
            rnd = int(race["round"])
            print(f"  -> Round {rnd}")

            # Defaults (race-level)
            rain_flag = 0
            track_temp_avg = 30.0

            try:
                session = fastf1.get_session(year, rnd, "R")
                session.load(telemetry=False, weather=True)
                laps = session.laps
                drivers = session.drivers

                # Weather
                w_data = session.weather_data
                if w_data is not None and "Rainfall" in w_data.columns:
                    try:
                        rain_flag = 1 if bool(w_data["Rainfall"].fillna(0).astype(float).gt(0).any()) else 0
                    except Exception:
                        rain_flag = 0

                if w_data is not None and "TrackTemp" in w_data.columns:
                    try:
                        t = pd.to_numeric(w_data["TrackTemp"], errors="coerce").mean()
                        track_temp_avg = 30.0 if pd.isna(t) else float(t)
                    except Exception:
                        track_temp_avg = 30.0

            except Exception as e:
                print(" ⚠️ FastF1 Error:", e)
                continue

            for d in drivers:
                drv_laps = laps.pick_drivers([d])
                if drv_laps.empty:
                    continue

                drv_info = session.get_driver(d)

                given_name = drv_info.get("FirstName", "") or ""
                family_name = drv_info.get("LastName", "") or ""

                # CLEAN AIR PACE (top 25% of clean laps)
                try:
                    clean_laps = drv_laps.pick_track_status('1').pick_wo_box().pick_quicklaps()
                except Exception:
                    clean_laps = drv_laps.pick_wo_box().pick_quicklaps()

                clean_times = clean_laps["LapTime"].dt.total_seconds().dropna()

                if clean_times.empty:
                    clean_air_pace = np.nan
                else:
                    quantile_25 = int(len(clean_times) * 0.25)
                    if quantile_25 < 1:
                        clean_air_pace = clean_times.mean()
                    else:
                        clean_air_pace = clean_times.nsmallest(quantile_25).mean()

                # GLOBAL PACE
                all_times = drv_laps["LapTime"].dt.total_seconds().dropna()
                if all_times.empty:
                    continue

                avg_pace_val = all_times.mean()
                best_lap_val = all_times.min()

                team_name = drv_info.get("TeamName", "") or ""

                entry = {
                    "year": int(year),
                    "round": int(rnd),
                    "DriverNumber": d,
                    "DriverKey": make_driver_key(given_name, family_name),
                    "DriverName": drv_info.get("FullName", "") or "",
                    "Team": team_name,
                    "TeamKey": make_team_key(team_name),
                    "clean_air_pace": clean_air_pace,
                    "avg_race_pace": avg_pace_val,
                    "is_rainy": int(rain_flag),
                    "track_temp": float(track_temp_avg),
                    "best_lap": best_lap_val
                }

                # PIT STOPS / PIT LOSS
                pit_mask = drv_laps["PitOutTime"].notna() | drv_laps["PitInTime"].notna()
                entry["pitstops_count"] = int(pit_mask.sum())

                median_pace = all_times.median()
                pit_losses = (all_times[pit_mask] - median_pace).clip(lower=0)
                entry["mean_pit_loss"] = (pit_losses.mean() if not pit_losses.empty else np.nan)

                # STINTS
                stints = drv_laps["Stint"].dropna().astype(int)
                stint_ids = sorted(stints.unique())
                entry["stint_count"] = int(len(stint_ids))

                def stint_compound(stint_id):
                    subset = drv_laps[drv_laps["Stint"] == stint_id]
                    compounds = subset["Compound"].dropna().unique()
                    return compounds[0] if len(compounds) else None

                for idx, stint_id in enumerate(stint_ids[:3], start=1):
                    subset = drv_laps[drv_laps["Stint"] == stint_id]
                    entry[f"stint{idx}_length"] = int(len(subset))
                    entry[f"stint{idx}_avg"] = subset["LapTime"].dt.total_seconds().mean()
                    entry[f"stint{idx}_compound"] = stint_compound(stint_id)

                for idx in range(len(stint_ids) + 1, 4):
                    entry[f"stint{idx}_length"] = np.nan
                    entry[f"stint{idx}_avg"] = np.nan
                    entry[f"stint{idx}_compound"] = None

                entry["compound_first_stint"] = entry.get("stint1_compound")

                # COMPOUND CHANGES
                compounds_clean = drv_laps["Compound"].ffill()
                entry["compound_changes"] = int(compounds_clean.ne(compounds_clean.shift()).sum() - 1)

                # GLOBAL DEGRADATION
                try:
                    first_avg = entry.get("stint1_avg")
                    last_avg = entry.get(f"stint{entry['stint_count']}_avg")
                    entry["degradation_global"] = (
                        (last_avg - first_avg) if first_avg is not None and last_avg is not None else np.nan
                    )
                except Exception:
                    entry["degradation_global"] = np.nan

                # CONSISTENCY
                entry["long_run_consistency"] = all_times.std()

                all_entries.append(entry)

            time.sleep(5)

    if not all_entries:
        print("⚠️ No FastF1 data extracted.")
        return

    df_new = pd.DataFrame(all_entries)
    df_new = ensure_keys(df_new)

    if os.path.exists(EXTRA_CSV_PATH):
        df_old = pd.read_csv(EXTRA_CSV_PATH)
        df_old = ensure_keys(df_old)
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(EXTRA_CSV_PATH, index=False)
    print(f"✔️ FastF1 enrichment → {EXTRA_CSV_PATH}")

# -------------------------------------------------------------------
# LOAD + UTILS
# -------------------------------------------------------------------

def load_data():
    if not os.path.exists(RESULTS_CSV_PATH):
        print("File not found.")
        return None

    df = pd.read_csv(RESULTS_CSV_PATH)
    df = ensure_keys(df)

    # inject circuitId via calendar
    if os.path.exists(CALENDAR_CSV_PATH):
        cal = pd.read_csv(CALENDAR_CSV_PATH)
        if "circuitId" not in df.columns:
            df = df.merge(cal[["year", "round", "circuitId"]], on=["year", "round"], how="left")
    else:
        print("⚠️ Calendar missing, running update_calendar()...")
        update_calendar()
        cal = pd.read_csv(CALENDAR_CSV_PATH)
        df = df.merge(cal[["year", "round", "circuitId"]], on=["year", "round"], how="left")

    # merge sprint data
    sprint_cols = ["sprint_pos", "sprint_grid", "sprint_points"]
    if os.path.exists(SPRINT_CSV_PATH):
        try:
            df_sprint = pd.read_csv(SPRINT_CSV_PATH)
            df_sprint = ensure_keys(df_sprint)
            cols_to_merge = ["year", "round", "DriverKey"] + [c for c in sprint_cols if c in df_sprint.columns]
            df = df.merge(df_sprint[cols_to_merge], on=["year", "round", "DriverKey"], how="left")
        except Exception as e:
            print(f"⚠️ Error loading sprint data: {e}")
            for c in sprint_cols:
                df[c] = np.nan
    else:
        for c in sprint_cols:
            df[c] = np.nan

    # safety: backfill TeamKey after merges
    df = ensure_keys(df)
    return df

def load_extra_features():
    if not os.path.exists(EXTRA_CSV_PATH):
        return None
    extra = pd.read_csv(EXTRA_CSV_PATH)
    extra = ensure_keys(extra)
    return extra

def get_rounds_for_race(race_name_keyword):
    if not os.path.exists(CALENDAR_CSV_PATH):
        print("Calendar not found.\nDownloading...")
        update_calendar()

    df = pd.read_csv(CALENDAR_CSV_PATH)
    filtered = df[df["raceName"].str.contains(race_name_keyword, case=False, na=False)]

    if filtered.empty:
        print(f'No race found with name "{race_name_keyword}".')
        return {}, None

    return dict(zip(filtered["year"], filtered["round"])), filtered.iloc[0]["raceName"]

def get_race_participants(df, year, rnd):
    r = df[(df["year"] == year) & (df["round"] == rnd)].sort_values("grid")
    if not r.empty:
        cols = ["DriverKey", "Team", "TeamKey"]
        if "grid" in r.columns:
            cols.append("grid")
        cols = [c for c in cols if c in r.columns]
        out = r[cols].drop_duplicates()
        return ensure_keys(out)

    return pd.DataFrame()

def has_real_qualifying(year: int, rnd: int) -> bool:
    if not os.path.exists(QUALI_CSV_PATH):
        return False
    try:
        df_q = pd.read_csv(QUALI_CSV_PATH)
        if "year" not in df_q.columns or "round" not in df_q.columns:
            return False
        mask = (df_q["year"] == year) & (df_q["round"] == rnd)
        return not df_q[mask].empty
    except:
        return False
