import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def train_and_predict(df, target_year, target_round, gp_name, use_real_grid=False):
    # Affichage du mode (pour que tu saches ce qui se passe)
    mode_str = "VRAIE GRILLE (Si dispo)" if use_real_grid else "GRILLE PRÉDITE (Full IA)"
    print(f"\n--- MACHINE LEARNING : {gp_name} ({target_year}) ---")
    print(f"⚙️  Mode : {mode_str}")

    # on prend les données des pilotes sans DNF
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()
    
    # on prépare l'encodage de tous les pilotes
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    # on combine tous les pilotes connus (passés et présents) pour que l'encodeur les connaisse tous
    # (ajout de .astype(str) par sécurité)
    all_drivers = pd.concat([df_clean["DriverName"], df[df["year"] == target_year]["DriverName"]]).astype(str).unique()
    all_teams = pd.concat([df_clean["Team"], df[df["year"] == target_year]["Team"]]).astype(str).unique()
    
    # on lance l'encodage 
    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    
    # on applique l'encodage sur le dataframe propre
    df_clean["driver_id"] = le_driver.transform(df_clean["DriverName"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean["Team"].astype(str))
    
    # on sépare temporellement les données (on apprend sur le passé)
    # Le filtre : année d'avant ou année en cours mais courses passées
    mask_train = (df_clean["year"] < target_year) | ((df_clean["year"] == target_year) & (df_clean["round"] < target_round))
    df_train = df_clean[mask_train]

    # MODELE 1 : prédiction de la qualif
    features_qualif = ["team_id", "driver_id", "year"]
    X_qualif = df_train[features_qualif]
    y_qualif = df_train["grid"]  # notre cible : la position sur la grille
    
    # apprentissage du modèle 1
    model_qualif = RandomForestRegressor(n_estimators=100, random_state=42)
    model_qualif.fit(X_qualif, y_qualif)
    
    # MODELE 2 : prédiction de la course
    features_race = ["grid", "team_id", "driver_id", "year"]
    X_race = df_train[features_race]
    y_race = df_train["position"]  # notre cible : la position à l'arrivée
    
    # apprentissage du modèle 2
    model_race = RandomForestRegressor(n_estimators=100, random_state=42)
    model_race.fit(X_race, y_race)
    print("   -> Modèles entraînés.")

    # la simulation finale
    # NOUVEAUTÉ : on regarde d'abord si on a la liste EXACTE des participants dans les données
    # AJOUT V1.2 : On récupère aussi la colonne 'grid' pour pouvoir l'utiliser
    official_entry_list = df[
        (df["year"] == target_year) & 
        (df["round"] == target_round)
    ].sort_values("grid")

    if not official_entry_list.empty:
        # on prend exactement ceux qui étaient là (avec leur grille si dispo)
        # On vérifie si la colonne grid existe bien avant de la sélectionner
        cols_to_keep = ["DriverName", "Team"]
        if "grid" in official_entry_list.columns:
            cols_to_keep.append("grid")
            
        drivers_to_simulate = official_entry_list[cols_to_keep].drop_duplicates()
        print(f"   -> Liste officielle trouvée ({len(drivers_to_simulate)} pilotes).")
    else:
        # on récupère les pilotes de la saison actuelle (logique de repli)
        print("   -> Course future (pas de data). Utilisation de la dernière grille connue.")
        df_current = df[df["year"] == target_year].sort_values("round", ascending=False)
        
        # sécurité début de saison
        if df_current.empty:
             df_current = df[df["year"] == target_year - 1].sort_values("round", ascending=False)
             
        drivers_to_simulate = df_current.drop_duplicates(subset=["DriverName"])[["DriverName", "Team"]]

    # lancement de la simulation
    simulation_results = []
    
    for _, row in drivers_to_simulate.iterrows():
        driver = row["DriverName"]
        team = row["Team"]
        
        try:
            d_id = le_driver.transform([str(driver)])[0]
            t_id = le_team.transform([str(team)])[0]
            
            # prédiction qualif
            X_pred_qualif = pd.DataFrame(
                [[t_id, d_id, target_year]],
                columns=["team_id", "driver_id", "year"]
            )

            # Etape 1 : on prédit la qualif
            pred_grid_raw = model_qualif.predict(X_pred_qualif)[0]
            
            # LOGIQUE V1.2 : Choix de la grille pour la course
            # Par défaut, on prend la prédiction de l'IA
            grid_input = pred_grid_raw
            
            # Si l'utilisateur veut la VRAIE grille et qu'on l'a dans les données
            if use_real_grid and "grid" in row and not pd.isna(row["grid"]):
                grid_input = row["grid"]

            # Etape 2 : on utilise la grille choisie pour prédire la course
            X_pred_race = pd.DataFrame(
                [[grid_input, t_id, d_id, target_year]],
                columns=["grid", "team_id", "driver_id", "year"]
            )
            pred_race_raw = model_race.predict(X_pred_race)[0]
            
            simulation_results.append({
                "Pilote": driver,
                "Ecurie": team,
                "Qualif_Score": pred_grid_raw,
                "Course_Score": pred_race_raw,
                "Grid_Used": grid_input # Pour info
            })
        except Exception:
            continue
            
    # mise en forme des résultats
    results = pd.DataFrame(simulation_results)
    
    # on trie tout pour avoir un classement propre
    # d'abord la grille
    results = results.sort_values(by="Qualif_Score")
    results["Grille_Predite"] = range(1, len(results) + 1)
    
    # ensuite la course
    results = results.sort_values(by="Course_Score")
    results["Position_Predite"] = range(1, len(results) + 1)
    
    # calcul du delta (position gagnée / perdue)
    results["Delta"] = results["Grille_Predite"] - results["Position_Predite"]
    
    # affichage des résultats
    print("\nRÉSULTATS DE LA SIMULATION :")
    # (Correction du print : le .to_string() doit être DANS le print)
    print(results[["Position_Predite", "Pilote", "Ecurie", "Grille_Predite", "Delta"]].head(20).to_string(index=False))
    
    print("\nAnalyse des facteurs clés :")
    imp_q = model_qualif.feature_importances_
    print(f"[Qualif] L'écurie compte pour {imp_q[0]*100:.0f}% vs Pilote {imp_q[1]*100:.0f}%")
    
    imp_r = model_race.feature_importances_
    print(f"[Course] La grille compte pour {imp_r[0]*100:.0f}% du résultat final.")