import pandas as pd
import numpy as np
import os
from src.data_manager import load_extra_features

# ---------------------------------------------------------
# 1) Driver recent form (last 3 GPs)
# ---------------------------------------------------------

def add_dual_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["year", "round"])
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    df["form_grid"] = df.groupby("DriverKey")["grid"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["form_race"] = df.groupby("DriverKey")["position"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    df["form_grid"] = df["form_grid"].fillna(13.0)
    df["form_race"] = df["form_race"].fillna(13.0)
    return df

# ---------------------------------------------------------
# 2) Circuit importance
# ---------------------------------------------------------

def add_circuit_impact(df: pd.DataFrame) -> pd.DataFrame:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    calendar_path = os.path.join(os.path.dirname(current_dir), "data", "races_calendar.csv")

    if not os.path.exists(calendar_path):
        df["circuit_importance"] = 0.5
        return df

    calendar = pd.read_csv(calendar_path)

    if "circuitId" not in df.columns:
        df = df.merge(calendar[["year", "round", "circuitId"]], on=["year", "round"], how="left")
    else:
        # Nettoyage doublons si merge précédent
        if "circuitId_x" in df.columns:
            df["circuitId"] = df["circuitId_x"].fillna(df["circuitId_y"])
            df = df.drop(columns=["circuitId_x", "circuitId_y"])

    df["circuitId"] = df["circuitId"].fillna("unknown")

    finishers = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)]

    if finishers.empty:
        df["circuit_importance"] = 0.5
        return df

    try:
        # Avoid data leakage: compute correlation using only historical data up to the PREVIOUS race
        # Use `df` to get all races including future ones to predict
        races = df[["year", "round", "circuitId"]].drop_duplicates()
        importances = []

        for _, race in races.iterrows():
            y = race["year"]
            r = race["round"]
            cid = race["circuitId"]

            # Historical data for THIS circuit BEFORE this race
            past_data = finishers[
                (finishers["circuitId"] == cid) &
                ((finishers["year"] < y) | ((finishers["year"] == y) & (finishers["round"] < r)))
            ]

            # Require at least 10 historical finisher samples to compute correlation reliably
            if len(past_data) >= 10:
                c = past_data["grid"].corr(past_data["position"])
                imp = float(c) if pd.notna(c) else 0.5
            else:
                imp = 0.5

            importances.append({
                "year": y,
                "round": r,
                "circuitId": cid,
                "circuit_importance": imp
            })

        df_importances = pd.DataFrame(importances)

        df = df.merge(df_importances, on=["year", "round", "circuitId"], how="left")
        df["circuit_importance"] = df["circuit_importance"].fillna(0.5)
    except Exception as e:
        print(f"⚠️ Erreur lors du calcul de circuit_importance: {e}")
        df["circuit_importance"] = 0.5

    return df

# ---------------------------------------------------------
# 3) FastF1 features (robust handling)
# ---------------------------------------------------------

def add_fastf1_features(df: pd.DataFrame) -> pd.DataFrame:
    extra = load_extra_features()
    fastf1_cols = ["avg_race_pace", "clean_air_pace", "best_lap", "pitstops_count", "mean_pit_loss", "is_rainy", "track_temp"]

    # If no extra file, create empty columns (0.0)
    if extra is None or extra.empty:
        df["avg_race_pace"] = 0.0
        df["clean_air_pace"] = 0.0
        df["best_lap"] = 0.0
        df["pitstops_count"] = 0
        df["mean_pit_loss"] = 0.0
        df["is_rainy"] = 0
        df["track_temp"] = 30.0
        return df

    # Clean mean_pit_loss if necessary
    if "pit_losses" in extra.columns and "mean_pit_loss" not in extra.columns:
        def clean_pit(val):
            try:
                if isinstance(val, str):
                    v = val.replace("[","").replace("]","").split(",")
                    nums = [float(x) for x in v if x.strip()]
                    return np.mean(nums) if nums else np.nan
                return float(val)
            except: return np.nan
        extra["mean_pit_loss"] = extra["pit_losses"].apply(clean_pit)

    # Merge
    # Keep only columns that actually exist in extra
    cols_to_merge = ["year", "round", "DriverKey"] + [c for c in fastf1_cols if c in extra.columns]
    # Safety: ensure DriverKey exists in df
    if "DriverKey" not in df.columns:
        df["DriverKey"] = df["DriverName"].str.lower()

    df = df.merge(extra[cols_to_merge], on=["year","round","DriverKey"], how="left", suffixes=("", "_extra"))

    # Fill NaNs (Median or 0) + types
    for c in ["avg_race_pace", "clean_air_pace", "best_lap", "mean_pit_loss"]:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(0.0 if pd.isna(med) else med).astype(float)
        else:
            df[c] = 0.0

    if "pitstops_count" in df.columns:
        df["pitstops_count"] = df["pitstops_count"].fillna(0).astype(int)
    else:
        df["pitstops_count"] = 0

    if "is_rainy" in df.columns:
        df["is_rainy"] = df["is_rainy"].fillna(0).astype(int)
    else:
        df["is_rainy"] = 0

    if "track_temp" in df.columns:
        medt = df["track_temp"].median()
        df["track_temp"] = df["track_temp"].fillna(30.0 if pd.isna(medt) else medt).astype(float)
    else:
        df["track_temp"] = 30.0

    return df


# ---------------------------------------------------------
# 4) SPRINT FEATURES
# ---------------------------------------------------------

def add_sprint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds contextual features from sprint results
    Strategy: additive (sprint = context, not target)
    """
    # safety : if sprint data was not merged
    required_cols = {"sprint_pos", "sprint_grid"}
    if not required_cols.issubset(df.columns):
        df["has_sprint"] = 0
        df["sprint_pos"] = df["grid"]
        df["sprint_grid"] = df["grid"]
        df["sprint_delta"] = 0.0
        return df

    # flag sprint weekends
    df["has_sprint"] = df["sprint_pos"].notna().astype(int)
    # sprint delta (gained/lost positions)
    df["sprint_delta"] = df["sprint_grid"] - df["sprint_pos"]
    # neutral fill for non sprint weekends
    df["sprint_delta"] = df["sprint_delta"].fillna(0.0)
    # if no sprint, assume sprint_pos == grid (neutral & realistic)
    df["sprint_pos"] = df["sprint_pos"].fillna(df["grid"])

    return df

# ---------------------------------------------------------
# 5) Career History (Advanced Stats)
# ---------------------------------------------------------

def add_driver_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "round"])
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    # Safety: ensure FastF1 columns exist before transform
    for c in ["avg_race_pace", "clean_air_pace", "best_lap", "mean_pit_loss", "is_rainy"]:
        if c not in df.columns: df[c] = 0.0

    grp = df.groupby("DriverKey")

    # Classic stats
    df["career_grid_avg"] = grp["grid"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_race_avg"] = grp["position"].transform(lambda x: x.shift(1).expanding().mean())

    # FastF1 stats
    df["career_race_pace"] = grp["avg_race_pace"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_clean_air_pace"] = grp["clean_air_pace"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_best_lap"] = grp["best_lap"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_pit_loss"] = grp["mean_pit_loss"].transform(lambda x: x.shift(1).expanding().mean())
    # v1.8 : wet skill (average positions gained under rainy races (grid -> finish), past-only)
    df["wet_gain"] = np.where(df["is_rainy"] == 1, (df["grid"] - df["position"]), np.nan)
    df["career_wet_skill"] = grp["wet_gain"].transform(lambda x: x.shift(1).expanding().mean())

    # Circuit stats
    if "circuitId" in df.columns:
        grpc = df.groupby(["DriverKey", "circuitId"])
        df["circuit_grid_skill"] = grpc["grid"].transform(lambda x: x.shift(1).expanding().mean())
        df["circuit_race_skill"] = grpc["position"].transform(lambda x: x.shift(1).expanding().mean())
    else:
        df["circuit_grid_skill"] = np.nan
        df["circuit_race_skill"] = np.nan

    # Fill missing values
    cols_fill = ["career_grid_avg", "career_race_avg", "circuit_grid_skill", "circuit_race_skill"]
    df[cols_fill] = df[cols_fill].fillna(14.0)

    cols_fill_f1 = ["career_race_pace", "career_clean_air_pace", "career_best_lap", "career_pit_loss", "career_wet_skill"]
    for c in cols_fill_f1:
        global_med = df[c].median()
        global_default = 0.0 if pd.isna(global_med) else global_med
        per_driver_med = df.groupby("DriverKey")[c].transform("median")
        df[c] = df[c].fillna(per_driver_med).fillna(global_default)

    df["pace_rank_season"] = (df.groupby(["year"])["career_race_pace"].rank(method="dense"))
    df["pace_rank_season"] = (df.groupby("year")["pace_rank_season"].transform(lambda x: x / x.max()))
    df["pace_rank_season"] = df["pace_rank_season"].fillna(0.5)

    return df
