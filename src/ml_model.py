import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants


# V1.5 : fonction pour calculer la forme récente des pilotes (qualif ET course) sur les 3 derniers rounds
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


# fonction pour préparer les données et les encodeurs
def encode_data(df):
    # on prend les données des pilotes sans les DNF
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()

    # on prépare l'encodage de tous les pilotes
    le_driver = LabelEncoder()
    le_team = LabelEncoder()

    # on combine tous les pilotes connus pour l'encodage
    all_drivers = df["DriverName"].astype(str).unique()
    all_teams = df["Team"].astype(str).unique()

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)

    # on applique l'encodage
    df_clean["driver_id"] = le_driver.transform(df_clean["DriverName"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean["Team"].astype(str))

    return df_clean, le_driver, le_team


# fonction pour entraîner les modèles
def train_models(df_train):
    # Paramètres de RandomForestRegressor (testés et optimisés)
    params = {
        "n_estimators": 600,
        "max_depth": 14,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42
    }

    # MODELE 1 v1.5 : prédiction de la qualif (avec maintenant la forme récente de qualif)
    features_qualif = ["team_id", "driver_id", "year", "form_grid"]
    model_qualif = RandomForestRegressor(**params)
    model_qualif.fit(df_train[features_qualif], df_train["grid"])
    # MODELE 2 v1.5 : prédiction de la course (avec maintenant la forme récente de course)
    features_race = ["grid", "team_id", "driver_id", "year", "form_race"]
    model_race = RandomForestRegressor(**params)
    model_race.fit(df_train[features_race], df_train["position"])

    return model_qualif, model_race


# fonction pour prédire une liste de pilotes avec leur historique de forme
def predict_race_outcome(models, drivers_df, year, target_round, le_driver, le_team, full_df, use_real_grid=False):
    model_qualif, model_race = models
    simulation_results = []
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
    # on lance la simulation pilote par pilote
    for _, row in drivers_df.iterrows():
        driver = row["DriverName"]
        team = row["Team"]

        # on récupère les formes
        forms = last_forms.get(driver, {"form_grid": 13.0, "form_race": 15.0})

        try:
            d_id = le_driver.transform([str(driver)])[0]
            t_id = le_team.transform([str(team)])[0]

            # Etape 1 on prédit la qualif
            X_q = pd.DataFrame(
                [[t_id, d_id, year, forms["form_grid"]]],
                columns=["team_id", "driver_id", "year", "form_grid"]
            )
            pred_grid = model_qualif.predict(X_q)[0]
            
            # choix de la grille
            grid_input = pred_grid
            if use_real_grid and "grid" in row and not pd.isna(row["grid"]):
                grid_input = row["grid"]
            
            # Etape 2 : on utilise la grille choisie pour prédire la course
            X_r = pd.DataFrame(
                [[grid_input, t_id, d_id, year, forms["form_race"]]],
                columns=["grid", "team_id", "driver_id", "year", "form_race"]
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
    # calcul des formes récentes
    df = add_dual_form(df)

    # encodage global
    df_clean, le_driver, le_team = encode_data(df)

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
    results = predict_race_outcome(models, target_list, target_year, target_round, le_driver, le_team, df, use_real_grid)

    # mise en forme intelligente
    results = results.sort_values("Grid_Input")
    results["Grille"] = range(1, len(results) + 1)

    results = results.sort_values("Course_Score")
    results["Pos"] = range(1, len(results) + 1)
        
    results["Delta"] = results["Grille"] - results["Pos"]
    results = results.sort_values("Pos")

    print("\nRÉSULTATS DE LA SIMULATION :")
    print(results[["Pos", "Pilote", "Ecurie", "Grille", "Delta"]].head(20).to_string(index=False))
