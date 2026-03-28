import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants, has_real_qualifying, load_real_qualifying, load_extra_features
from src.models.qualif_ranker import QualifRankerLGBM


warnings.filterwarnings("ignore", message="Mean of empty slice")

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

# ---------------------------------------------------------
# 6) Encoding data
# ---------------------------------------------------------

def encode_data(df: pd.DataFrame) -> (pd.DataFrame, LabelEncoder, LabelEncoder, LabelEncoder):
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()

    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()

    team_col = "TeamKey" if "TeamKey" in df.columns else "Team"

    all_drivers = df["DriverKey"].astype(str).unique()
    all_teams = df[team_col].astype(str).unique()

    if "circuitId" in df.columns:
        all_circuits = df["circuitId"].astype(str).unique()
    else:
        all_circuits = ["unknown"]
        df_clean["circuitId"] = "unknown"

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    le_circuit.fit(all_circuits)

    df_clean["driver_id"] = le_driver.transform(df_clean["DriverKey"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean[team_col].astype(str))
    df_clean["circuit_id"] = le_circuit.transform(df_clean["circuitId"].astype(str))

    return df_clean, le_driver, le_team, le_circuit

# ---------------------------------------------------------
# 7) Model training
# ---------------------------------------------------------

def train_models(df_train: pd.DataFrame) -> (QualifRankerLGBM, RandomForestRegressor):
    # RandomForest hyperparameters
    params_qualif = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        # best parameters with tuning
        "n_estimators": 83,
        "learning_rate": 0.010417146488237577,
        "num_leaves": 60,
        "max_depth": -1,
        "min_child_samples": 26,
        "subsample": 0.9962990060021659,
        "colsample_bytree": 0.8896856637603093,
        "reg_lambda": 2.679269781861703,
        "reg_alpha": 0.7714673192056071
    }
    params_race = {
        "n_estimators": 320,
        "max_depth": 13,
        "min_samples_split": 13,
        "min_samples_leaf": 5,
        "max_features": None,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    }

    # Qualifying features
    features_qualif = [
        "team_id", "driver_id", "year", 
        "form_grid", "circuit_importance", "circuit_id", 
        "career_grid_avg", "circuit_grid_skill"
    ]
    # Filtering to keep only existing features
    features_qualif = [f for f in features_qualif if f in df_train.columns]
    
    model_qualif = QualifRankerLGBM(params=params_qualif)
    model_qualif.fit(df_train, features_qualif, target_col="grid")

    # Race features
    features_race = [
        "grid",
        "form_race",
        "career_race_avg",
        "pace_rank_season",
        "team_id", "driver_id", "year", 
        "circuit_importance", "circuit_id",
        "circuit_race_skill",
        "career_race_pace", "career_clean_air_pace", "career_best_lap", "career_pit_loss", "career_wet_skill",
        "has_sprint", "sprint_delta",
        "is_rainy", "track_temp"
    ]
    features_race = [f for f in features_race if f in df_train.columns]

    model_race = RandomForestRegressor(**params_race)
    model_race.fit(df_train[features_race], df_train["position"])

    return model_qualif, model_race

# ---------------------------------------------------------
# 8) Predictions
# ---------------------------------------------------------

def predict_race_outcome(models: (QualifRankerLGBM, RandomForestRegressor), drivers_df: pd.DataFrame, year: int, target_round: int, le_driver: LabelEncoder, le_team: LabelEncoder, le_circuit: LabelEncoder, full_df: pd.DataFrame, use_real_grid=False) -> pd.DataFrame:
    model_qualif, model_race = models
    simulation_results = []

    # -----------------------------
    # Helpers
    # -----------------------------
    def safe_le_transform(le, value, default=-1):
        """LabelEncoder safe transform (unknown -> default)."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        v = str(value)
        try:
            # fast membership
            if hasattr(le, "classes_") and v in set(le.classes_):
                return int(le.transform([v])[0])
        except Exception:
            pass
        return default

    def get_col(row, *candidates, default=None):
        for c in candidates:
            if c in row and pd.notna(row[c]):
                return row[c]
        return default

    # -----------------------------
    # 0) Name map + defaults
    # -----------------------------
    name_map = {}
    if "DriverName" in full_df.columns:
        name_map = (
            full_df.dropna(subset=["DriverName"])
            .drop_duplicates(subset=["DriverKey"])
            .set_index("DriverKey")["DriverName"]
            .to_dict()
        )

    default_race_pace = float(full_df["career_race_pace"].median()) if "career_race_pace" in full_df.columns else 95.0
    default_clean_air = float(full_df["career_clean_air_pace"].median()) if "career_clean_air_pace" in full_df.columns else 95.0
    default_best_lap  = float(full_df["career_best_lap"].median()) if "career_best_lap" in full_df.columns else 95.0
    default_pit_loss  = float(full_df["career_pit_loss"].median()) if "career_pit_loss" in full_df.columns else 25.0
    default_wet_skill = float(full_df["career_wet_skill"].median()) if "career_wet_skill" in full_df.columns else 0.0
    default_track_temp = float(full_df["track_temp"].median()) if "track_temp" in full_df.columns else 30.0

    # -----------------------------
    # 1) Circuit context
    # -----------------------------
    target_race_info = full_df[(full_df["year"] == year) & (full_df["round"] == target_round)]
    impact_val = float(target_race_info.iloc[0]["circuit_importance"]) if (not target_race_info.empty and "circuit_importance" in target_race_info.columns) else 0.5
    circuit_name_str = str(target_race_info.iloc[0]["circuitId"]) if (not target_race_info.empty and "circuitId" in target_race_info.columns) else "unknown"
    c_id = safe_le_transform(le_circuit, circuit_name_str, default=0)

    # -----------------------------
    # 2) Build driver stats map (optimized)
    # -----------------------------
    mask_hist = (full_df["year"] < year) | ((full_df["year"] == year) & (full_df["round"] < target_round))
    hist = full_df[mask_hist].sort_values(["year", "round"]).copy()

    last_by_driver = {}
    if not hist.empty:
        last_rows = hist.groupby("DriverKey", sort=False).tail(1)
        last_by_driver = last_rows.set_index("DriverKey").to_dict(orient="index")

    def default_stats():
        return {
            "form_grid": 13.0,
            "form_race": 15.0,
            "career_grid_avg": 14.0,
            "career_race_avg": 14.0,
            "circuit_grid_skill": 14.0,
            "circuit_race_skill": 14.0,
            "career_race_pace": default_race_pace,
            "career_clean_air_pace": default_clean_air,
            "career_best_lap": default_best_lap,
            "career_pit_loss": default_pit_loss,
            "career_wet_skill": default_wet_skill,
            "pace_rank_season": 0.5
        }

    last_stats_map = {}
    for driver in drivers_df["DriverKey"].dropna().unique():
        s = default_stats()
        rowd = last_by_driver.get(driver)
        if rowd:
            for k in s.keys():
                if k in rowd and rowd[k] is not None and not (isinstance(rowd[k], float) and np.isnan(rowd[k])):
                    s[k] = rowd[k]
        last_stats_map[driver] = s

    # -----------------------------
    # 3) Batch Qualif prediction (Ranker)
    # -----------------------------
    qualif_rows = []
    driver_order = []

    for _, row in drivers_df.iterrows():
        driver = row.get("DriverKey")
        if pd.isna(driver):
            continue

        # encoding uses TeamKey if available, else Team
        team_key = get_col(row, "TeamKey", default=None)
        team_name = get_col(row, "Team", default=None)
        team_for_encoding = team_key if team_key is not None else team_name if team_name is not None else "unknown"

        stats = last_stats_map.get(driver)
        if stats is None:
            continue

        d_id = safe_le_transform(le_driver, driver, default=-1)
        t_id = safe_le_transform(le_team, team_for_encoding, default=-1)

        qualif_rows.append({
            "team_id": t_id,
            "driver_id": d_id,
            "year": year,
            "round": target_round,
            "form_grid": stats["form_grid"],
            "circuit_importance": impact_val,
            "circuit_id": c_id,
            "career_grid_avg": stats["career_grid_avg"],
            "circuit_grid_skill": stats["circuit_grid_skill"],
        })
        driver_order.append(driver)

    pred_grid_map = {}

    if qualif_rows:
        X_q_batch = pd.DataFrame(qualif_rows)

        # categorical types for LGBM ranker (safe)
        for c in ["team_id", "driver_id", "circuit_id"]:
            if c in X_q_batch.columns:
                X_q_batch[c] = X_q_batch[c].astype("category")

        # choose feature list expected by ranker
        feats = getattr(model_qualif, "feature_names", None)
        if not feats:
            feats = X_q_batch.columns.tolist()
        feats = [f for f in feats if f in X_q_batch.columns]

        try:
            ranks = model_qualif.predict(X_q_batch, feats)  # expects ranker wrapper signature
            for drv, rk in zip(driver_order, ranks):
                pred_grid_map[drv] = int(rk)
        except Exception as e:
            print(f"⚠️ Erreur Ranker Qualif: {e}")

            # fallback intelligent: sort by (form_grid, career_grid_avg)
            tmp = X_q_batch.copy()
            tmp["_drv"] = driver_order
            tmp = tmp.sort_values(["form_grid", "career_grid_avg"], ascending=True)
            for i, drv in enumerate(tmp["_drv"].tolist(), start=1):
                pred_grid_map[drv] = i

    # -----------------------------
    # 4) Race prediction loop
    # -----------------------------
    # Use model expected feature names if available (sklearn)
    race_expected = getattr(model_race, "feature_names_in_", None)

    for _, row in drivers_df.iterrows():
        driver = row.get("DriverKey")
        if pd.isna(driver):
            continue

        stats = last_stats_map.get(driver)
        if stats is None:
            continue

        team_key = get_col(row, "TeamKey", default=None)
        team_name = get_col(row, "Team", default=None)
        team_for_encoding = team_key if team_key is not None else team_name if team_name is not None else "unknown"
        team_display = team_name if team_name is not None else (str(team_key).replace("_", " ").title() if team_key else "Unknown")

        nice_name = name_map.get(driver, row.get("DriverName", str(driver)))

        d_id = safe_le_transform(le_driver, driver, default=-1)
        t_id = safe_le_transform(le_team, team_for_encoding, default=-1)

        # grid choice: AI by default, real if requested + available
        ai_grid_pos = pred_grid_map.get(driver, 10)
        grid_input = ai_grid_pos
        if use_real_grid and "grid" in row and pd.notna(row["grid"]) and row["grid"] > 0:
            grid_input = float(row["grid"])

        has_sprint_val = int(row.get("has_sprint", 0)) if pd.notna(row.get("has_sprint", 0)) else 0
        s_delta = float(row.get("sprint_delta", 0.0)) if pd.notna(row.get("sprint_delta", 0.0)) else 0.0
        is_rainy_val = int(row.get("is_rainy", 0)) if pd.notna(row.get("is_rainy", 0)) else 0
        track_temp_val = float(row.get("track_temp", default_track_temp)) if pd.notna(row.get("track_temp", default_track_temp)) else float(default_track_temp)

        X_r = pd.DataFrame([{
            "grid": grid_input,
            "form_race": stats["form_race"],
            "career_race_avg": stats["career_race_avg"],
            "pace_rank_season": stats["pace_rank_season"],
            "team_id": t_id,
            "driver_id": d_id,
            "year": year,
            "circuit_importance": impact_val,
            "circuit_id": c_id,
            "circuit_race_skill": stats["circuit_race_skill"],
            "career_race_pace": stats["career_race_pace"],
            "career_clean_air_pace": stats["career_clean_air_pace"],
            "career_best_lap": stats["career_best_lap"],
            "career_pit_loss": stats["career_pit_loss"],
            "career_wet_skill": stats["career_wet_skill"],
            "has_sprint": has_sprint_val,
            "sprint_delta": s_delta,
            "is_rainy": is_rainy_val,
            "track_temp": track_temp_val,
        }])

        # ensure we feed exactly what the model expects
        if race_expected is not None:
            cols = [c for c in race_expected if c in X_r.columns]
            X_use = X_r[cols]
        else:
            X_use = X_r

        try:
            pred_race = float(model_race.predict(X_use)[0])
        except Exception:
            continue

        simulation_results.append({
            "DriverKey": driver,
            "DriverName": nice_name,
            "Team": team_display,
            "Course_Score": pred_race,
            "Grid_Input": grid_input
        })

    return pd.DataFrame(simulation_results)

# ---------------------------------------------------------
# 9) MAIN FUNCTION
# ---------------------------------------------------------

def train_and_predict(df: pd.DataFrame, target_year: int, target_round: int, gp_name: str, use_real_grid=False) -> None:
    print(f"\n--- MACHINE LEARNING : {gp_name} ({target_year}) ---")
    
    df = df.copy()
    df = df[(df["year"] < target_year) | ((df["year"] == target_year) & (df["round"] <= target_round))]
    # 1) Enrichment
    df = add_dual_form(df)
    df = add_circuit_impact(df)
    df = add_fastf1_features(df)
    df = add_sprint_features(df)
    df = add_driver_history(df)

    # 2) Encoding
    df_clean, le_driver, le_team, le_circuit = encode_data(df)

    # 3) Split
    mask_train = (df_clean["year"] < target_year) | ((df_clean["year"] == target_year) & (df_clean["round"] < target_round))
    df_train = df_clean[mask_train]

    models = train_models(df_train)
    print("   -> Models trained.")

    # 4) Grid
    target_list = get_race_participants(df, target_year, target_round)

    # weather management
    ctx_cols = ["DriverKey", "has_sprint", "sprint_delta", "is_rainy", "track_temp"]
    existing = [c for c in ctx_cols if c in df.columns]
    if "DriverKey" in existing and len(existing) > 1:
        race_ctx = df[(df["year"] == target_year) & (df["round"] == target_round)][existing].drop_duplicates()
        target_list = target_list.merge(race_ctx, on="DriverKey", how="left")

    for c, default in [("has_sprint", 0), ("sprint_delta", 0.0), ("is_rainy", 0), ("track_temp", 30.0)]:
        if c in target_list.columns:
            target_list[c] = target_list[c].fillna(default)


    # Real grid management
    has_grid_in_main = "grid" in target_list.columns and target_list["grid"].notna().any()
    has_grid_in_latest = has_real_qualifying(target_year, target_round)

    if use_real_grid:
        if has_grid_in_main:
            pass
        elif has_grid_in_latest:
            target_list = load_real_qualifying(target_year, target_round)
        else:
            print("❗Real grid unavailable. Switching to AI grid mode.")
            use_real_grid = False
    
    if target_list.empty:
        print("❌ Error: participant list is empty")
        return

    # 5) Prediction
    results = predict_race_outcome(
        models, target_list, target_year, target_round,
        le_driver, le_team, le_circuit, df, use_real_grid
    )
    
    if results.empty:
        print("❌ Error: no prediction generated.")
        return

    # 6) Display
    results = results.sort_values("Grid_Input")
    results["Grid"] = range(1, len(results) + 1)
    results = results.sort_values("Course_Score")
    results["Pos"] = range(1, len(results) + 1)
    results["Delta"] = results["Grid"] - results["Pos"]
    results = results.sort_values("Pos")

    print("\nSIMULATION RESULTS:")
    print(results[["Pos", "DriverName", "Team", "Grid", "Delta"]].head(22).to_string(index=False))


