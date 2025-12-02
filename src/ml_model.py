import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants


# V1.5-beta : fonction pour calculer la forme récente des pilotes (qualif ET course) sur les 3 derniers rounds
def add_dual_form(df):
    # on trie  par ordre chronologique (histoire de récupérer les "3 derniers")
    df = df.sort_values(by=["year", "round"])
    # on s'assure qu'on a bien des chiffres
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    # on calcule la forme récente en qualifications : shift(1) => on décale d'une ligne vers le bas pour avoir la course du passé
    # window=3 => on prend une fenêtre de 3 courses
    df["form_grid"] = df.groupby("DriverName")["grid"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    # On calcule la forme récente en course
    df["form_race"] = df.groupby("DriverName")["position"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    # on remplit les trous (débuts de carrière) par 13.0 (milieu de peloton)
    df["form_grid"] = df["form_grid"].fillna(13.0)
    df["form_race"] = df["form_race"].fillna(13.0)

    return df


# V1.5 : fonction pour ajouter l'impact du circuit (via races_calendar.csv)
def add_circuit_impact(df):
    # on va utiliser l'importance historique de la qualif pour chaque circuit
    # en utilisant notre race_calendar.csv pour identifier chaque circuit
    # on charge le calendrier :
    current_dir = os.path.dirname(os.path.abspath(__file__))
    calendar_path = os.path.join(os.path.dirname(current_dir), "data", "races_calendar.csv")
    if not os.path.exists(calendar_path):
        print("Calendrier introuvable.")
        df["circuit_importance"] = 0.6
        return df
    calendar = pd.read_csv(calendar_path)
    # On fusionne le tout pour ajouter circuitId aux données de course
    df_merged = df.merge(calendar[["year", "round", "circuitId"]], on=["year", "round"], how="left")
    # on ne prend que les pilotes qui ont fini une course
    df_finishers = df_merged[df_merged["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)]
    # on calcule la corrélation grille/position pour chaque circuitId
    circuit_stats = df_finishers.groupby("circuitId")[["grid", "position"]].corr().iloc[0::2, -1]
    # on nettoie les résultats, en les transformant en dictionnaire simple
    circuit_impact_map = {}
    for idx, val in circuit_stats.items():
        # idx est un tuple (circuitId, "grid")
        circuit_id = idx[0]
        if not pd.isna(val):
            circuit_impact_map[circuit_id] = val
        else:
            circuit_impact_map[circuit_id] = 0.5   # valeur par défaut si pas de données
    # on rempace df par la version fusionnée
    df = df_merged
    # on map la valeur
    df["circuit_importance"] = df["circuitId"].map(circuit_impact_map)
    # si on a pas de valeur, alors valeur par défaut
    df["circuit_importance"] = df["circuit_importance"].fillna(0.5)

    return df


# V1.5 : fonction pour ajouter la prise en compte des résultats de la carrière d'un pilote
def add_driver_history(df):
    """
    On calcule 4 indicateurs historiques :
    - career_grid_avg : Moyenne Qualif en carrière
    - career_race_avg : Moyenne Course en carrière
    - circuit_grid_skill : Moyenne Qualif sur CE circuit
    - circuit_race_skill : Moyenne Course sur CE circuit
    """
    df = df.sort_values(by=["year", "round"])
    # Carrière globale (qualif et course)
    grouped_driver = df.groupby("DriverName")
    df["career_grid_avg"] = grouped_driver["grid"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["career_race_avg"] = grouped_driver["position"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    # force sur le circuit
    grouped_circuit = df.groupby(["DriverName", "circuitId"])
    df["circuit_grid_skill"] = grouped_circuit["grid"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["circuit_race_skill"] = grouped_circuit["position"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    # remplissage par défaut, (14.0 => milieu/bas de grille)
    cols_to_fill = ["career_grid_avg", "career_race_avg", "circuit_grid_skill", "circuit_race_skill"]
    df[cols_to_fill] = df[cols_to_fill].fillna(14.0)

    return df


# fonction pour préparer les données et les encodeurs
def encode_data(df):
    # on prend les données des pilotes sans les DNF
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()

    # on prépare l'encodage de tous les pilotes, des écuries et des circuits
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()

    # on combine tous les pilotes, écuries et circuits connus pour l'encodage
    all_drivers = df["DriverName"].astype(str).unique()
    all_teams = df["Team"].astype(str).unique()
    all_circuits = df["circuitId"].astype(str).unique()

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    le_circuit.fit(all_circuits)

    # on applique l'encodage
    df_clean["driver_id"] = le_driver.transform(df_clean["DriverName"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean["Team"].astype(str))
    df_clean["circuit_id"] = le_circuit.transform(df_clean["circuitId"].astype(str))

    return df_clean, le_driver, le_team, le_circuit


# fonction pour entraîner les modèles
def train_models(df_train):
    # Paramètres de RandomForestRegressor (testés et optimisés)
    params_qualif = {
        "n_estimators": 400,
        "min_samples_split": 4,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "max_depth": 14,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42
    }
    params_race = {
        "n_estimators": 800,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "max_depth": 8,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42
    }

    # MODELE 1 v1.5 : prédiction de la qualif (avec maintenant la forme récente de qualif + l'historique)
    features_qualif = [
        "team_id",
        "driver_id",
        "year",
        "form_grid",
        "circuit_importance",
        "circuit_id",
        "career_grid_avg",
        "circuit_grid_skill"
        ]
    model_qualif = RandomForestRegressor(**params_qualif)
    model_qualif.fit(df_train[features_qualif], df_train["grid"])
    # MODELE 2 v1.5 : prédiction de la course (avec maintenant la forme récente de course + l'historique)
    features_race = [
        "grid",
        "team_id",
        "driver_id",
        "year",
        "form_race",
        "circuit_importance",
        "circuit_id",
        "career_race_avg",
        "circuit_race_skill"
        ]
    model_race = RandomForestRegressor(**params_race)
    model_race.fit(df_train[features_race], df_train["position"])

    return model_qualif, model_race


# fonction pour prédire une liste de pilotes avec leur historique de forme
def predict_race_outcome(models, drivers_df, year, target_round, le_driver, le_team, le_circuit, full_df, use_real_grid=False):
    model_qualif, model_race = models
    simulation_results = []
    last_stats_map = {}
    # on prépare un dico des formes actuelles de chaque pilote de la grille
    last_forms = {}
    for driver in drivers_df["DriverName"].unique():
        # on récupère l'historique complet du pilote
        driver_history = full_df[
            (full_df["DriverName"] == driver) &
            ((full_df["year"] < year) | ((full_df["year"] == year) & (full_df["round"] < target_round)))
        ]
        if not driver_history.empty:
            # on prend la ligne la plus récente (la dernière)
            last_stats = driver_history.iloc[-1]
            last_forms[driver] = {
                "form_grid": last_stats["form_grid"],
                "form_race": last_stats["form_race"]
            }
        else:
            # si c'est un nouveau pilote, moyenne par défaut
            last_forms[driver] = {"form_grid": 13.0, "form_race": 15.0}
    
    # On récupère l'impact circuit (trouver le circuitId de la course qu'on veut)
    target_race_info = full_df[(full_df["year"] == year) & (full_df["round"] == target_round)]
    impact_val = 0.5   # par défaut
    circuit_name_str = "unknown"
    if not target_race_info.empty:
        impact_val = target_race_info.iloc[0]["circuit_importance"]
        circuit_name_str = target_race_info.iloc[0]["circuitId"]
    else:
        # si c'est une course future, on cherche dans le calendrier (donc on l'ouvre encore)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        calendar_path = os.path.join(os.path.dirname(current_dir), "data", "races_calendar.csv")
        if os.path.exists(calendar_path):
            cal = pd.read_csv(calendar_path)
            # on cherche le circuit de cette année/round
            race_cal = cal[(cal["year"] == year) & (cal["round"] == target_round)]
            if not race_cal.empty:
                cid = race_cal.iloc[0]["circuitId"]
                # maintenant on cherche la donnée historique de ce circuit dans full_df
                if "circuitId" in full_df.columns:
                    hist_circuit = full_df[full_df["circuitId"] == cid]
                    if not hist_circuit.empty:
                        impact_val = hist_circuit["circuit_importance"].mean()
    
    # encodage du circuit cible
    try:
        c_id = le_circuit.transform([str(circuit_name_str)])[0]
    except:
        c_id = 0
    
    # on récupère les stats pilote
    for driver in drivers_df["DriverName"].unique():
        # historique global du pilote avant la course
        driver_history = full_df[
            (full_df["DriverName"] == driver) &
            ((full_df["year"] < year) | ((full_df["year"] == year) & (full_df["round"] < target_round)))
        ]
        # historique sur CE circuit
        driver_circuit_history = driver_history[driver_history["circuitId"] == circuit_name_str]
        # valeurs par défaut (rookies)
        stats = {
            "form_grid": 13.0, "form_race": 15.0,
            "career_grid_avg": 14.0, "career_race_avg": 14.0,
            "circuit_grid_skill": 14.0, "circuit_race_skill": 14.0
        }
        if not driver_history.empty:
            last = driver_history.iloc[-1]
            stats.update({
                "form_grid": last["form_grid"],
                "form_race": last["form_race"],
                "career_grid_avg": last["career_grid_avg"],
                "career_race_avg": last["career_race_avg"]
            })
        if not driver_circuit_history.empty:
            last_c = driver_circuit_history.iloc[-1]
            stats.update({
                "circuit_grid_skill": last_c["circuit_grid_skill"],
                "circuit_race_skill": last_c["circuit_race_skill"]
            })
        last_stats_map[driver] = stats

    # on lance la simulation pilote par pilote
    for _, row in drivers_df.iterrows():
        driver = row["DriverName"]
        team = row["Team"]
        # on récupère les formes
        stats = last_stats_map.get(driver)

        try:
            d_id = le_driver.transform([str(driver)])[0]
            t_id = le_team.transform([str(team)])[0]

            # Etape 1 on prédit la qualif
            X_q = pd.DataFrame(
                [[t_id, d_id, year, stats["form_grid"], impact_val, c_id, stats["career_grid_avg"], stats["circuit_grid_skill"]]],
                columns=["team_id", "driver_id", "year", "form_grid", "circuit_importance", "circuit_id", "career_grid_avg", "circuit_grid_skill"]
            )
            pred_grid = model_qualif.predict(X_q)[0]
            
            # choix de la grille
            grid_input = pred_grid
            if use_real_grid and "grid" in row and not pd.isna(row["grid"]):
                grid_input = row["grid"]
            
            # Etape 2 : on utilise la grille choisie pour prédire la course
            X_r = pd.DataFrame(
                [[grid_input, t_id, d_id, year, stats["form_race"], impact_val, c_id, stats["career_race_avg"], stats["circuit_race_skill"]]],
                columns=["grid", "team_id", "driver_id", "year", "form_race", "circuit_importance", "circuit_id", "career_race_avg", "circuit_race_skill"],
            )
            pred_race = model_race.predict(X_r)[0]

            simulation_results.append({
                "Pilote": driver,
                "Ecurie": team,
                "Qualif_Score": pred_grid,
                "Course_Score": pred_race,
                "Grid_Input": grid_input
            })
        except Exception:
            continue
    
    return pd.DataFrame(simulation_results)


# fonction principale (appelée par main.py)
def train_and_predict(df, target_year, target_round, gp_name, use_real_grid=False):
    # calcul des formes récentes, de l'importance du circuit et de l'historique
    df = add_dual_form(df)
    df = add_circuit_impact(df)
    df = add_driver_history(df)

    # encodage global
    df_clean, le_driver, le_team, le_circuit = encode_data(df)

    # séparation temporelle (on apprend sur le passé)
    mask_train = (df_clean["year"] < target_year) | ((df_clean["year"] == target_year) & (df_clean["round"] < target_round))
    df_train = df_clean[mask_train]

    # entraînement
    models = train_models(df_train)

    # récupération dynamique des participants
    target_list = get_race_participants(df, target_year, target_round)
    if target_list.empty:
        print("Erreur : impossible de trouver une liste de pilotes.")
        return

    # prédiction
    results = predict_race_outcome(models, target_list, target_year, target_round, le_driver, le_team, le_circuit, df, use_real_grid)

    # mise en forme intelligente
    results = results.sort_values("Grid_Input")
    results["Grille"] = range(1, len(results) + 1)

    results = results.sort_values("Course_Score")
    results["Pos"] = range(1, len(results) + 1)
        
    results["Delta"] = results["Grille"] - results["Pos"]
    results = results.sort_values("Pos")

    print("\nRÉSULTATS DE LA SIMULATION :")
    print(results[["Pos", "Pilote", "Ecurie", "Grille", "Delta"]].head(20).to_string(index=False))
