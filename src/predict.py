import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.models.qualif_ranker import QualifRankerLGBM

# ---------------------------------------------------------
# 1) Predictions
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
