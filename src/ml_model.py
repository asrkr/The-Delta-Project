import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants, has_real_qualifying, load_real_qualifying, load_extra_features


warnings.filterwarnings("ignore", message="Mean of empty slice")


# ---------------------------------------------------------
# 1) Forme récente pilote (3 derniers GP)
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
# 2) Importance du circuit
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
# 3) Features FastF1 (Gestion Robustesse)
# ---------------------------------------------------------
def add_fastf1_features(df):
    extra = load_extra_features()
    fastf1_cols = ["avg_race_pace", "best_lap", "pitstops_count", "mean_pit_loss"]

    # Si pas de fichier extra, on crée les colonnes vides (0.0)
    if extra is None or extra.empty:
        for c in fastf1_cols: df[c] = 0.0
        return df

    # Nettoyage mean_pit_loss si nécessaire
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
    # On ne garde que les colonnes qui existent vraiment dans extra
    cols_to_merge = ["year", "round", "DriverKey"] + [c for c in fastf1_cols if c in extra.columns]
    # Sécurité : si DriverKey absent côté df
    if "DriverKey" not in df.columns:
        df["DriverKey"] = df["DriverName"].str.lower()

    df = df.merge(extra[cols_to_merge], on=["year","round","DriverKey"], how="left", suffixes=("", "_extra"))

    # Remplissage des NaN (Médiane ou 0)
    for c in fastf1_cols:
        if c in df.columns:
            med = df[c].median()
            val = 0.0 if pd.isna(med) else med
            df[c] = df[c].fillna(val)
        else:
            df[c] = 0.0
            
    return df


# ---------------------------------------------------------
# 4) Historique Carrière (Stats Avancées)
# ---------------------------------------------------------
def add_driver_history(df):
    df = df.sort_values(["year", "round"])
    
    # Sécurité : on s'assure que les colonnes FastF1 existent avant de faire le transform
    for c in ["avg_race_pace", "best_lap", "mean_pit_loss"]:
        if c not in df.columns: df[c] = 0.0

    grp = df.groupby("DriverKey")
    
    # Stats classiques
    df["career_grid_avg"] = grp["grid"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_race_avg"] = grp["position"].transform(lambda x: x.shift(1).expanding().mean())
    
    # Stats FastF1
    df["career_race_pace"] = grp["avg_race_pace"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_best_lap"] = grp["best_lap"].transform(lambda x: x.shift(1).expanding().mean())
    df["career_pit_loss"] = grp["mean_pit_loss"].transform(lambda x: x.shift(1).expanding().mean())

    # Stats Circuit
    if "circuitId" in df.columns:
        grpc = df.groupby(["DriverKey", "circuitId"])
        df["circuit_grid_skill"] = grpc["grid"].transform(lambda x: x.shift(1).expanding().mean())
        df["circuit_race_skill"] = grpc["position"].transform(lambda x: x.shift(1).expanding().mean())
    else:
        df["circuit_grid_skill"] = np.nan
        df["circuit_race_skill"] = np.nan

    # Remplissage
    cols_fill = ["career_grid_avg", "career_race_avg", "circuit_grid_skill", "circuit_race_skill"]
    df[cols_fill] = df[cols_fill].fillna(14.0)
    
    cols_fill_f1 = ["career_race_pace", "career_best_lap", "career_pit_loss"]
    for c in cols_fill_f1:
        med = df[c].median()
        val = 0.0 if pd.isna(med) else med
        df[c] = df[c].fillna(df.groupby("DriverKey")[c].transform("median"))
    
    df["expected_finish_from_grid"] = (
        df.groupby("circuitId")["grid"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["expected_finish_from_grid"] = df["expected_finish_from_grid"].fillna(df["grid"])

    df["grid_delta"] = df["grid"] - df["expected_finish_from_grid"]

    grid_mean = df.groupby("circuitId")["grid"].transform("mean")
    grid_std = df.groupby("circuitId")["grid"].transform("std").replace(0, 1)

    df["grid_z"] = (df["grid"] - grid_mean) / grid_std
    df["grid_z"] = df["grid_z"].fillna(0.0)

    return df


# ---------------------------------------------------------
# 5) Encodage
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
# 6) Entraînement
# ---------------------------------------------------------
def train_models(df_train):
    # Paramètres (Tu peux remettre les tiens ici)
    params_qualif = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1
    }
    params_race = {
        "n_estimators": 1200,
        "max_depth": 6,
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1
    }

    # Liste Qualif
    features_qualif = [
        "team_id", "driver_id", "year", 
        "form_grid", "circuit_importance", "circuit_id", 
        "career_grid_avg", "circuit_grid_skill"
    ]
    # Filtre pour ne garder que ce qui existe
    features_qualif = [f for f in features_qualif if f in df_train.columns]
    
    model_qualif = RandomForestRegressor(**params_qualif)
    model_qualif.fit(df_train[features_qualif], df_train["grid"])

    # Liste Course (AVEC LES FEATURES FASTF1)
    features_race = [
        "grid",
        "grid_z",
        "grid_delta",
        "team_id", "driver_id", "year", 
        "form_race", "circuit_importance", "circuit_id", 
        "career_race_avg", "circuit_race_skill",
        "career_race_pace", "career_best_lap", "career_pit_loss" # <-- Ici
    ]
    features_race = [f for f in features_race if f in df_train.columns]

    model_race = RandomForestRegressor(**params_race)
    model_race.fit(df_train[features_race], df_train["position"])

    # === Feature Importances ===
    
    qualif_importances = get_feature_importances(model_qualif, features_qualif)
    race_importances = get_feature_importances(model_race, features_race)

    print("\n📊 Qualifying Model — Feature Importances")
    print(qualif_importances.to_string(index=False))

    print("\n🏁 Race Model — Feature Importances")
    print(race_importances.to_string(index=False))
    

    return model_qualif, model_race


# ---------------------------------------------------------
# 7) Prédiction (TA VERSION CORRIGÉE)
# ---------------------------------------------------------
def predict_race_outcome(models, drivers_df, year, target_round, le_driver, le_team, le_circuit, full_df, use_real_grid=False):
    model_qualif, model_race = models
    simulation_results = []
    # -----------------------------
    # Valeurs par défaut
    # -----------------------------
    default_race_pace = (full_df["career_race_pace"].median() if "career_race_pace" in full_df else 95.0)
    default_best_lap = (full_df["career_best_lap"].median() if "career_best_lap" in full_df else 95.0)
    default_pit_loss = (full_df["career_pit_loss"].median() if "career_pit_loss" in full_df else 25.0)
    # -----------------------------
    # 1. Circuit ID + importance
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
    # 2. Récupération stats pilotes
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
            "career_pit_loss": default_pit_loss
        }

        if not history.empty:
            last = history.iloc[-1]
            for k in stats:
                if k in last and not pd.isna(last[k]):
                    stats[k] = last[k]

        last_stats_map[driver] = stats

    # -----------------------------
    # 3. Prédiction pilote par pilote
    # -----------------------------
    for _, row in drivers_df.iterrows():
        driver = row["DriverKey"]
        team = row["Team"]
        stats = last_stats_map.get(driver)

        if stats is None:
            continue

        try:
            d_id = le_driver.transform([driver])[0]
            t_id = le_team.transform([team])[0]

            # ===== QUALIF (si grille IA) =====
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

            # ===== ÉTAPE 1 — CONTEXTUALISATION DU GRID =====
            expected_finish = stats["career_grid_avg"]
            grid_delta = grid_input - expected_finish

            grid_mean = drivers_df["grid"].mean()
            grid_std = grid_std = full_df[
                (full_df["year"] < year) |
                ((full_df["year"] == year) & (full_df["round"] < target_round))
            ]["grid"].std()
            if pd.isna(grid_std) or grid_std == 0.0:
                grid_std == 1.0
            grid_z = (grid_input - grid_mean) / grid_std

            # ===== COURSE =====
            X_r = pd.DataFrame([[
                grid_input,
                grid_z,
                grid_delta,
                t_id, d_id, year,
                stats["form_race"],
                impact_val, c_id,
                stats["career_race_avg"],
                stats["circuit_race_skill"],
                stats["career_race_pace"],
                stats["career_best_lap"],
                stats["career_pit_loss"]
            ]], columns=[
                "grid",
                "grid_z",
                "grid_delta",
                "team_id", "driver_id", "year",
                "form_race",
                "circuit_importance", "circuit_id",
                "career_race_avg",
                "circuit_race_skill",
                "career_race_pace",
                "career_best_lap",
                "career_pit_loss"
            ])

            pred_race = model_race.predict(X_r)[0]

            simulation_results.append({
                "DriverKey": driver,
                "DriverName": row.get("DriverName", driver.title()),
                "Ecurie": team,
                "Course_Score": pred_race,
                "Grid_Input": grid_input
            })

        except Exception as e:
            # 👉 utile au debug, tu pourras retirer plus tard
            print(f"[PRED ERROR] {driver} → {e}")
            continue

    return pd.DataFrame(simulation_results)



# ---------------------------------------------------------
# 8) FONCTION PRINCIPALE
# ---------------------------------------------------------
def train_and_predict(df, target_year, target_round, gp_name, use_real_grid=False):
    print(f"\n--- MACHINE LEARNING : {gp_name} ({target_year}) ---")
    
    # 1) Enrichissement
    df = add_dual_form(df)
    df = add_circuit_impact(df)
    df = add_fastf1_features(df)
    df = add_driver_history(df)

    # 2) Encodage
    df_clean, le_driver, le_team, le_circuit = encode_data(df)

    # 3) Split
    mask_train = (df_clean["year"] < target_year) | ((df_clean["year"] == target_year) & (df_clean["round"] < target_round))
    df_train = df_clean[mask_train]

    models = train_models(df_train)
    print("   -> Modèles entraînés.")

    # 4) Grille
    target_list = get_race_participants(df, target_year, target_round)

    # Gestion grille réelle
    has_grid_in_main = "grid" in target_list.columns and target_list["grid"].notna().any()
    has_grid_in_latest = has_real_qualifying(target_year, target_round)

    if use_real_grid:
        if has_grid_in_main:
            pass
        elif has_grid_in_latest:
            target_list = load_real_qualifying(target_year, target_round)
        else:
            print("❗Grille réelle indisponible. Passage en mode Grille IA.")
            use_real_grid = False
    
    if target_list.empty:
        print("❌ Erreur : Liste des participants vide.")
        return

    # 5) Prédiction
    results = predict_race_outcome(
        models, target_list, target_year, target_round,
        le_driver, le_team, le_circuit, df, use_real_grid
    )
    
    if results.empty:
        print("❌ Erreur : Aucune prédiction générée.")
        return

    # 6) Affichage
    results = results.sort_values("Grid_Input")
    results["Grille"] = range(1, len(results) + 1)
    results = results.sort_values("Course_Score")
    results["Pos"] = range(1, len(results) + 1)
    results["Delta"] = results["Grille"] - results["Pos"]
    results = results.sort_values("Pos")

    print("\nRÉSULTATS DE LA SIMULATION :")
    print(results[["Pos", "DriverName", "Ecurie", "Grille", "Delta"]].head(20).to_string(index=False))


def get_feature_importances(model, feature_names):
    """
    Retourne un DataFrame trié des feature importances d'un modèle sklearn
    """
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
