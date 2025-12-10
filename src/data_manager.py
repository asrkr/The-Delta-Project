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


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

# -------------------------------------------------------------------
# DRIVER KEY CREATION
# -------------------------------------------------------------------

def make_driver_key(given_name: str, family_name: str) -> str:
    if not given_name or not family_name:
        return None
    # Strip accents and special characters
    def normalize(s):
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-zA-Z]", "", s)
        return s.lower()
    g = normalize(given_name)
    f = normalize(family_name)
    if not g or not f:
        return None
    driver_key = f"{g[0]}_{f}"
    return driver_key

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
                
                df = pd.DataFrame(races[0]["Results"])
                
                # Safeguard with .get to prevent errors on older seasons
                if "Driver" in df.columns:
                    df["DriverKey"] = df["Driver"].apply(lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", "")))
                    df["DriverName"] = df["Driver"].apply(lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}".strip())
                if "Constructor" in df.columns:
                    df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
                df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
                df["position"] = pd.to_numeric(df["position"], errors="coerce")
                
                # Filter valid columns (including points)
                cols_ok = ["DriverKey", "DriverName", "Team", "grid", "position", "status", "points", "circuitId"]
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
        
        # Clean data extraction
        df["DriverKey"] = df["Driver"].apply(lambda x: make_driver_key(x.get("givenName", ""), x.get("familyName", "")))
        df["DriverName"] = df["Driver"].apply(lambda x: f"{x.get("givenName", "")} {x.get("familyName", "")}".strip())
        df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
        
        # IMPORTANT: Standardize column name for the rest of the pipeline
        df["grid"] = pd.to_numeric(df["position"], errors="coerce")
        
        # FIX: Add year and round to enable filtering later!
        df["year"] = year
        df["round"] = rnd
        
        return df[["DriverKey", "DriverName", "Team", "grid", "year", "round"]]
        
    except Exception as e:
        print(f"Erreur fetch quali : {e}")
        return None


def update_latest_qualifying(year, rnd):
    df = fetch_qualifying_results(year, rnd)
    if df is None:
        return False
    # Overwrite the file to keep only the latest request
    df.to_csv(QUALI_CSV_PATH, index=False)
    return True


def has_real_qualifying(year: int, rnd: int) -> bool:
    if not os.path.exists(QUALI_CSV_PATH):
        return False
    try:
        df_q = pd.read_csv(QUALI_CSV_PATH)
        # Check if the file contains the required columns
        if "year" not in df_q.columns or "round" not in df_q.columns:
            return False
        mask = (df_q["year"] == year) & (df_q["round"] == rnd)
        return not df_q[mask].empty
    except:
        return False


def load_real_qualifying(year, rnd):
    if not os.path.exists(QUALI_CSV_PATH):
        return pd.DataFrame()
    
    df_q = pd.read_csv(QUALI_CSV_PATH)
    
    mask = (df_q["year"] == year) & (df_q["round"] == rnd)
    quali = df_q[mask].copy()
    
    # Keep only useful columns
    cols = [c for c in ["DriverKey", "DriverName", "Team", "grid", "year", "round"] if c in quali.columns]
    return quali[cols]


# -------------------------------------------------------------------
# ERGAST — UPDATE WITH MERGE (incremental)
# -------------------------------------------------------------------

def update_database(start_year=2001, end_year=2025):
    print(f"📌 Updating Ergast results {start_year}-{end_year}.")
    all_races = []
    
    for year in range(start_year, end_year + 1):
        print(f" Saison {year}...", end=" ")
        cpt = 0
        for rnd in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/results.json"
            result = _fetch_race_result(url)
            
            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame):
                result["year"] = year
                result["round"] = rnd
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
        # Remove old data from the update range to avoid duplicates
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    # Sort and save
    if "grid" in df_final.columns:
        df_final = df_final.sort_values(["year", "round", "grid"])
    else:
        df_final = df_final.sort_values(["year", "round"])
        
    df_final.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"✔️ Saved → {RESULTS_CSV_PATH}.")


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

            try:
                session = fastf1.get_session(year, rnd, "R")
                session.load(telemetry=False, weather=False)
                laps = session.laps
                drivers = session.drivers
            except Exception as e:
                print(" ⚠️ FastF1 Error:", e)
                continue

            for d in drivers:
                drv_laps = laps.pick_drivers([d])
                if drv_laps.empty:
                    continue

                drv_info = session.get_driver(d)

                given_name = drv_info.get("FirstName", "")
                family_name = drv_info.get("LastName", "")

                entry = {
                    "year": year,
                    "round": rnd,
                    "DriverNumber": d,
                    "DriverKey": make_driver_key(given_name, family_name),
                    "DriverName": drv_info.get("FullName", ""),
                    "Team": drv_info.get("TeamName", "")
                }
                # =============================
                # GLOBAL PACE
                # =============================
                lap_times = drv_laps["LapTime"].dt.total_seconds().dropna()
                if lap_times.empty:
                    continue
                entry["avg_race_pace"] = lap_times.mean()
                entry["best_lap"] = lap_times.min()
                # =============================
                # PIT STOPS / PIT LOSS 
                # =============================
                pit_mask = drv_laps["PitOutTime"].notna() | drv_laps["PitInTime"].notna()
                entry["pitstops_count"] = pit_mask.sum()

                median_pace = lap_times.median()
                pit_losses = (lap_times[pit_mask] - median_pace).clip(lower=0)

                entry["mean_pit_loss"] = (
                    pit_losses.mean() if not pit_losses.empty else np.nan
                )
                # =============================
                # STINTS
                # =============================
                stints = drv_laps["Stint"].dropna().astype(int)
                stint_ids = sorted(stints.unique())
                entry["stint_count"] = len(stint_ids)

                def stint_compound(stint_id):
                    subset = drv_laps[drv_laps["Stint"] == stint_id]
                    compounds = subset["Compound"].dropna().unique()
                    return compounds[0] if len(compounds) else None

                for idx, stint_id in enumerate(stint_ids[:3], start=1):
                    subset = drv_laps[drv_laps["Stint"] == stint_id]
                    entry[f"stint{idx}_length"] = len(subset)
                    entry[f"stint{idx}_avg"] = subset["LapTime"].dt.total_seconds().mean()
                    entry[f"stint{idx}_compound"] = stint_compound(stint_id)

                # compléter les stints manquants
                for idx in range(len(stint_ids) + 1, 4):
                    entry[f"stint{idx}_length"] = np.nan
                    entry[f"stint{idx}_avg"] = np.nan
                    entry[f"stint{idx}_compound"] = None

                entry["compound_first_stint"] = entry.get("stint1_compound")
                # =============================
                # COMPOUND CHANGES
                # =============================
                compounds_clean = drv_laps["Compound"].ffill()
                entry["compound_changes"] = (
                    compounds_clean.ne(compounds_clean.shift()).sum() - 1
                )
                # =============================
                # GLOBAL DEGRADATION
                # =============================
                try:
                    first_avg = entry.get("stint1_avg")
                    last_avg = entry.get(f"stint{entry['stint_count']}_avg")
                    entry["degradation_global"] = (
                        last_avg - first_avg
                        if first_avg is not None and last_avg is not None
                        else np.nan
                    )
                except Exception:
                    entry["degradation_global"] = np.nan
                # =============================
                # CONSISTENCY
                # =============================
                entry["long_run_consistency"] = lap_times.std()
                all_entries.append(entry)

    # =============================
    # SAVE (INCREMENTAL)
    # =============================
    if not all_entries:
        print("⚠️ No FastF1 data extracted.")
        return

    df_new = pd.DataFrame(all_entries)

    if os.path.exists(EXTRA_CSV_PATH):
        df_old = pd.read_csv(EXTRA_CSV_PATH)
        df_old = df_old[
            (df_old["year"] < start_year) | (df_old["year"] > end_year)
        ]
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

    # --- AUTOMATIC INJECTION OF circuitId VIA CALENDAR ---
    if os.path.exists(CALENDAR_CSV_PATH):
        cal = pd.read_csv(CALENDAR_CSV_PATH)
        # Safe merge
        if "circuitId" not in df.columns:
            df = df.merge(
                cal[["year", "round", "circuitId"]],
                on=["year", "round"],
                how="left"
            )
    else:
        print("⚠️ Calendar missing, running update_calendar()...")
        update_calendar()
        cal = pd.read_csv(CALENDAR_CSV_PATH)
        df = df.merge(
            cal[["year", "round", "circuitId"]],
            on=["year", "round"],
            how="left"
        )

    return df


def load_extra_features():
    return pd.read_csv(EXTRA_CSV_PATH) if os.path.exists(EXTRA_CSV_PATH) else None


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
        # Return race participants found in history
        cols = ["DriverKey", "Team"]
        if "grid" in r.columns: cols.append("grid")
        return r[cols].drop_duplicates()
        
    return pd.DataFrame()
