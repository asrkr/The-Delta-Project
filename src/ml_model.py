import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import get_race_participants


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
    # MODELE 1 : prédiction de la qualif
    features_qualif = ["team_id", "driver_id", "year"]
    model_qualif = RandomForestRegressor(n_estimators=600, max_depth=14, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)
    model_qualif.fit(df_train[features_qualif], df_train["grid"])
    # MODELE 2 : prédiction de la course
    features_race = ["grid", "team_id", "driver_id", "year"]
    model_race = RandomForestRegressor(n_estimators=600, max_depth=14, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)
    model_race.fit(df_train[features_race], df_train["position"])
    return model_qualif, model_race


# fonction pour prédire une liste de pilotes
def predict_race_outcome(models, drivers_df, year, le_driver, le_team, use_real_grid=False):
    model_qualif, model_race = models
    simulation_results = []

    for _, row in drivers_df.iterrows():
        driver = row["DriverName"]
        team = row["Team"]

        try:
            d_id = le_driver.transform([str(driver)])[0]
            t_id = le_team.transform([str(team)])[0]

            # Etape 1 on prédit la qualif
            X_q = pd.DataFrame(
                [[t_id, d_id, year]],
                columns=["team_id", "driver_id", "year"]
            )
            pred_grid = model_qualif.predict(X_q)[0]
            
            # choix de la grille
            grid_input = pred_grid
            if use_real_grid and "grid" in row and not pd.isna(row["grid"]):
                grid_input = row["grid"]
            
            # Etape 2 : on utilise la grille choisie pour prédire la course
            X_r = pd.DataFrame(
                [[grid_input, t_id, d_id, year]],
                columns=["grid", "team_id", "driver_id", "year"]
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
    results = predict_race_outcome(models, target_list, target_year, le_driver, le_team, use_real_grid)

    # mise en forme intelligente
    results = results.sort_values("Grid_Input")
    results["Grille"] = range(1, len(results) + 1)

    results = results.sort_values("Course_Score")
    results["Pos"] = range(1, len(results) + 1)
        
    results["Delta"] = results["Grille"] - results["Pos"]
    results = results.sort_values("Pos")

    print("\nRÉSULTATS DE LA SIMULATION :")
    print(results[["Pos", "Pilote", "Ecurie", "Grille", "Delta"]].head(20).to_string(index=False))
