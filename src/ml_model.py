import pandas as pd
import warnings
from src.data_manager import get_race_participants, has_real_qualifying, load_real_qualifying

# Imports for the refactored modules
from src.features import (
    add_dual_form,
    add_circuit_impact,
    add_fastf1_features,
    add_sprint_features,
    add_driver_history,
)
from src.train import encode_data, train_models
from src.predict import predict_race_outcome

warnings.filterwarnings("ignore", message="Mean of empty slice")

# ---------------------------------------------------------
# 1) MAIN ORCHESTRATOR FUNCTION
# ---------------------------------------------------------

def train_and_predict(df: pd.DataFrame, target_year: int, target_round: int, gp_name: str, use_real_grid=False) -> None:
    print(f"\n--- MACHINE LEARNING : {gp_name} ({target_year}) ---")
    
    df = df.copy()
    df = df[(df["year"] < target_year) | ((df["year"] == target_year) & (df["round"] <= target_round))]

    # 1) Enrichment (from features.py)
    df = add_dual_form(df)
    df = add_circuit_impact(df)
    df = add_fastf1_features(df)
    df = add_sprint_features(df)
    df = add_driver_history(df)

    # 2) Encoding (from train.py)
    df_clean, le_driver, le_team, le_circuit = encode_data(df)

    # 3) Split and Train (from train.py)
    mask_train = (df_clean["year"] < target_year) | ((df_clean["year"] == target_year) & (df_clean["round"] < target_round))
    df_train = df_clean[mask_train]

    models = train_models(df_train)
    print("   -> Models trained.")

    # 4) Grid Preparation
    target_list = get_race_participants(df, target_year, target_round)

    # Weather management
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

    # 5) Prediction (from predict.py)
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

