import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants, has_real_qualifying, load_real_qualifying, load_extra_features


warnings.filterwarnings("ignore", message="Mean of empty slice")

# ---------------------------------------------------------
# 1) Driver recent form (last 3 GPs)
# ---------------------------------------------------------

def add_dual_form(df):
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

def add_circuit_impact(df):
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
        corr = finishers.groupby("circuitId")[["grid", "position"]].corr().iloc[0::2, -1]
        # Reconstruction propre du dictionnaire
        impact_map = {}
        for idx, val in corr.items():
            circuit_id = idx[0] # idx est un tuple (circuitId, 'grid')
            impact_map[circuit_id] = val if not pd.isna(val) else 0.5
            
        df["circuit_importance"] = df["circuitId"].map(impact_map).fillna(0.5)
    except:
        df["circuit_importance"] = 0.5

    return df

# ---------------------------------------------------------
# 3) FastF1 features (robust handling)
# ---------------------------------------------------------

def add_fastf1_features(df):
    extra = load_extra_features()
    fastf1_cols = ["avg_race_pace", "best_lap", "pitstops_count", "mean_pit_loss"]

    # If no extra file, create empty columns (0.0)
    if extra is None or extra.empty:
        for c in fastf1_cols: df[c] = 0.0
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

    # Fill NaNs (Median or 0)
    for c in fastf1_cols:
        if c in df.columns:
            med = df[c].median()
            val = 0.0 if pd.isna(med) else med
            df[c] = df[c].fillna(val)
        else:
            df[c] = 0.0
            
    return df

# ---------------------------------------------------------
# 4) SPRINT FEATURES
# ---------------------------------------------------------

def add_sprint_features(df):
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

def add_driver_history(df):
    df = df.sort_values(["year", "round"])
    
    # Safety: ensure FastF1 columns exist before transform
    for c in ["avg_race_pace", "best_lap", "mean_pit_loss"]:
        if c not in df.columns: df[c] = 0.0

    grp = df.groupby("DriverKey")
    
    # Classic stats
    df["career_grid_avg"] = grp["grid"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_race_avg"] = grp["position"].transform(lambda x: x.shift(1).expanding().mean())
    
    # FastF1 stats
    df["career_race_pace"] = grp["avg_race_pace"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_best_lap"] = grp["best_lap"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_pit_loss"] = grp["mean_pit_loss"].transform(lambda x: x.shift(1).expanding().mean())

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
    
    cols_fill_f1 = ["career_race_pace", "career_best_lap", "career_pit_loss"]
    for c in cols_fill_f1:
        med = df[c].median()
        val = 0.0 if pd.isna(med) else med
        df[c] = df[c].fillna(df.groupby("DriverKey")[c].transform("median"))

    df["pace_rank_season"] = (df.groupby(["year"])["career_race_pace"].rank(method="dense"))
    df["pace_rank_season"] = (df.groupby("year")["pace_rank_season"].transform(lambda x: x / x.max()))
    df["pace_rank_season"] = df["pace_rank_season"].fillna(0.5)

    return df

# ---------------------------------------------------------
# 6) Encoding data
# ---------------------------------------------------------

def encode_data(df):
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()

    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()

    all_drivers = df["DriverKey"].astype(str).unique()
    all_teams = df["Team"].astype(str).unique()
    if "circuitId" in df.columns:
        all_circuits = df["circuitId"].astype(str).unique()
    else:
        all_circuits = ["unknown"]
        df_clean["circuitId"] = "unknown"

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    le_circuit.fit(all_circuits)

    df_clean["driver_id"] = le_driver.transform(df_clean["DriverKey"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean["Team"].astype(str))
    df_clean["circuit_id"] = le_circuit.transform(df_clean["circuitId"].astype(str))

    return df_clean, le_driver, le_team, le_circuit


# ---------------------------------------------------------
# 7) Model training
# ---------------------------------------------------------
def train_models(df_train):
    # RandomForest hyperparameters
    params_qualif = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 6,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1
    }
    params_race = {
        "n_estimators": 200,
        "max_depth": 18,
        "min_samples_split": 6,
        "min_samples_leaf": 8,
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
    
    model_qualif = RandomForestRegressor(**params_qualif)
    model_qualif.fit(df_train[features_qualif], df_train["grid"])

    # Race features
    features_race = [
        "grid",
        "form_race",
        "career_race_avg",
        "pace_rank_season",
        "team_id", "driver_id", "year", 
        "circuit_importance", "circuit_id",
        "circuit_race_skill",
        "career_race_pace", "career_best_lap", "career_pit_loss",
        "has_sprint", "sprint_delta"
    ]
    features_race = [f for f in features_race if f in df_train.columns]

    model_race = RandomForestRegressor(**params_race)
    model_race.fit(df_train[features_race], df_train["position"])

    
    # Feature Importances - OPTIONAL, ONLY FOR IMPROVEMENTS/DEBUTS
    """
    qualif_importances = get_feature_importances(model_qualif, features_qualif)
    race_importances = get_feature_importances(model_race, features_race)
    
    print("\n📊 Qualifying Model — Feature Importances")
    print(qualif_importances.to_string(index=False))

    print("\n🏁 Race Model — Feature Importances")
    print(race_importances.to_string(index=False))
    """

    return model_qualif, model_race

# ---------------------------------------------------------
# 8) Predictions
# ---------------------------------------------------------

def predict_race_outcome(models, drivers_df, year, target_round, le_driver, le_team, le_circuit, full_df, use_real_grid=False):
    model_qualif, model_race = models
    simulation_results = []
    # -----------------------------
    # Helper: Driver Name Map
    # -----------------------------
    name_map = {}
    if "DriverName" in full_df.columns:
        # We drop duplicates to keep the mapping unique
        name_map = full_df.dropna(subset=["DriverName"]).set_index("DriverKey")["DriverName"].to_dict()
    # -----------------------------
    # Default values
    # -----------------------------
    default_race_pace = (full_df["career_race_pace"].median() if "career_race_pace" in full_df else 95.0)
    default_best_lap = (full_df["career_best_lap"].median() if "career_best_lap" in full_df else 95.0)
    default_pit_loss = (full_df["career_pit_loss"].median() if "career_pit_loss" in full_df else 25.0)
    # -----------------------------
    # 1. Circuit ID + Importance
    # -----------------------------
    target_race_info = full_df[(full_df["year"] == year) & (full_df["round"] == target_round)]
    impact_val = 0.5
    circuit_name_str = "unknown"

    if not target_race_info.empty:
        impact_val = target_race_info.iloc[0]["circuit_importance"]
        circuit_name_str = target_race_info.iloc[0]["circuitId"]

    try:
        c_id = le_circuit.transform([str(circuit_name_str)])[0]
    except Exception:
        c_id = 0

    # -----------------------------
    # 2. Retrieve driver stats
    # -----------------------------
    last_stats_map = {}

    for driver in drivers_df["DriverKey"].unique():
        history = full_df[
            (full_df["DriverKey"] == driver) &
            (
                (full_df["year"] < year) |
                ((full_df["year"] == year) & (full_df["round"] < target_round))
            )
        ]

        stats = {
            "form_grid": 13.0,
            "form_race": 15.0,
            "career_grid_avg": 14.0,
            "career_race_avg": 14.0,
            "circuit_grid_skill": 14.0,
            "circuit_race_skill": 14.0,
            "career_race_pace": default_race_pace,
            "career_best_lap": default_best_lap,
            "career_pit_loss": default_pit_loss,
            "pace_rank_season": 0.5
        }

        if not history.empty:
            last = history.iloc[-1]
            for k in stats:
                if k in last and not pd.isna(last[k]):
                    stats[k] = last[k]

        last_stats_map[driver] = stats
    # -----------------------------
    # 3. Driver-by-driver prediction
    # -----------------------------
    for _, row in drivers_df.iterrows():
        driver = row["DriverKey"]
        team = row["Team"]
        stats = last_stats_map.get(driver)

        if stats is None:
            continue
        
        # DISPLAY FIX: Get the nice name from our map, fallback to row, then to key
        nice_name = name_map.get(driver, row.get("DriverName", driver.title()))

        try:
            d_id = le_driver.transform([driver])[0]
            t_id = le_team.transform([team])[0]

            # ===== QUALIFYING (if AI grid) =====
            X_q = pd.DataFrame([[
                t_id, d_id, year,
                stats["form_grid"], impact_val, c_id,
                stats["career_grid_avg"], stats["circuit_grid_skill"]
            ]], columns=[
                "team_id", "driver_id", "year",
                "form_grid", "circuit_importance", "circuit_id",
                "career_grid_avg", "circuit_grid_skill"
            ])

            pred_grid = model_qualif.predict(X_q)[0]

            grid_input = pred_grid
            if use_real_grid and "grid" in row and not pd.isna(row["grid"]):
                grid_input = row["grid"]
            
            # Retrieve sprint data if present
            has_sprint_val = row["has_sprint"] if "has_sprint" in row else 0
            s_delta = row["sprint_delta"] if "sprint_delta" in row and pd.notna(row["sprint_delta"]) else 0.0

            # ===== RACE =====
            X_r = pd.DataFrame([[
                grid_input,
                stats["form_race"],
                stats["career_race_avg"],
                stats["pace_rank_season"],
                t_id, d_id, year,
                impact_val, c_id,
                stats["circuit_race_skill"],
                stats["career_race_pace"],
                stats["career_best_lap"],
                stats["career_pit_loss"],
                has_sprint_val,
                s_delta
            ]], columns=[
                "grid",
                "form_race",
                "career_race_avg",
                "pace_rank_season",
                "team_id", "driver_id", "year",
                "circuit_importance", "circuit_id",
                "circuit_race_skill",
                "career_race_pace",
                "career_best_lap",
                "career_pit_loss",
                "has_sprint", "sprint_delta"
            ])

            pred_race = model_race.predict(X_r)[0]

            simulation_results.append({
                "DriverKey": driver,
                "DriverName": nice_name, # <--- Used the fixed name here
                "Team": team,
                "Course_Score": pred_race,
                "Grid_Input": grid_input
            })

        except Exception as e:
            # 👉 useful for debugging, can be removed later
            print(f"[PRED ERROR] {driver} → {e}")
            continue

    return pd.DataFrame(simulation_results)

# ---------------------------------------------------------
# 9) MAIN FUNCTION
# ---------------------------------------------------------

def train_and_predict(df, target_year, target_round, gp_name, use_real_grid=False):
    print(f"\n--- MACHINE LEARNING : {gp_name} ({target_year}) ---")
    
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
    print(results[["Pos", "DriverName", "Team", "Grid", "Delta"]].head(20).to_string(index=False))


# ---------------------------------------------------------
# TOOLS FOR BENCHMARKS
# ---------------------------------------------------------


def get_feature_importances(model, feature_names):
    """
    Return a sorted DataFrame of a sklearn model's feature importances
    """
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def add_sprint_features_test(df):
    df["has_sprint"] = 0
    df["sprint_delta"] = 0.0
    return df
