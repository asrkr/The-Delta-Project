import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.data_manager import load_data
import time

def run_simulation(season_to_simulate, use_real_grid=False):
    # Affichage du mode choisi pour confirmation
    mode_str = "VRAIE GRILLE (Analyse)" if use_real_grid else "GRILLE PRÉDITE (Full IA)"
    print(f"\n🎬 --- DÉMARRAGE DE LA SIMULATION : SAISON {season_to_simulate} ---")
    print(f"⚙️  Mode activé : {mode_str}")
    
    # 1. Chargement des données complètes
    df = load_data()
    if df is None: return

    # On nettoie d'abord les données globales (pour l'encodage)
    df_clean = df[df['status'].str.contains('Finished|Lap|Lapped', regex=True, na=False)].copy()

    # ENCODAGE GLOBAL
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    # Conversion en string pour sécurité
    all_drivers = df['DriverName'].astype(str).unique()
    all_teams = df['Team'].astype(str).unique()
    
    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    
    # On applique l'encodage
    df_clean['driver_id'] = le_driver.transform(df_clean['DriverName'].astype(str))
    df_clean['team_id'] = le_team.transform(df_clean['Team'].astype(str))

    # 2. IDENTIFIER LES COURSES DE LA SAISON CIBLE
    races_in_season = df[df['year'] == season_to_simulate].sort_values('round')['round'].unique()
    
    if len(races_in_season) == 0:
        print(f"❌ Pas de données pour la saison {season_to_simulate}.")
        return

    score_card = [] # Pour stocker nos victoires/défaites

    # 3. BOUCLE TEMPORELLE (Course par course)
    for race_round in races_in_season:
        print(f"\n🏁 Round {race_round} en cours de traitement...", end="") # petit end="" pour garder la ligne propre
        
        # --- A. ENTRAÎNEMENT (LE PASSÉ) ---
        mask_train = (df_clean['year'] < season_to_simulate) | \
                     ((df_clean['year'] == season_to_simulate) & (df_clean['round'] < race_round))
        
        train_data = df_clean[mask_train]
        
        if len(train_data) < 100:
            print("   ⚠️ Pas assez de données historiques, on saute.")
            continue

        # Modèle QUALIF
        # On définit explicitement les colonnes pour s'en souvenir après
        cols_qualif = ['team_id', 'driver_id', 'year']
        model_qualif = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model_qualif.fit(train_data[cols_qualif], train_data['grid'])

        # Modèle COURSE
        cols_race = ['grid', 'team_id', 'driver_id', 'year']
        model_race = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model_race.fit(train_data[cols_race], train_data['position'])

        # --- B. RÉALITÉ DU JOUR (LE PRÉSENT) ---
        current_race_data = df_clean[
            (df_clean['year'] == season_to_simulate) & 
            (df_clean['round'] == race_round)
        ].copy()
        
        if current_race_data.empty: continue

        # On récupère le VRAI vainqueur pour vérifier après
        vrai_vainqueur = current_race_data.sort_values('position').iloc[0]['DriverName']

        # --- C. PRÉDICTION ---
        predictions = []
        for _, row in current_race_data.iterrows():
            d_id = row['driver_id']
            t_id = row['team_id']
            
            # --- CORRECTION ICI : On utilise des DataFrames ---
            
            # 1. On prédit la grille (même si on ne l'utilise pas, c'est bien de l'avoir)
            X_q = pd.DataFrame([[t_id, d_id, season_to_simulate]], columns=cols_qualif)
            pred_grid = model_qualif.predict(X_q)[0]
            
            # CHOIX STRATÉGIQUE : Quelle grille donner au modèle de course ?
            if use_real_grid:
                grid_input = row['grid'] # La vraie grille officielle
            else:
                grid_input = pred_grid   # La grille devinée par l'IA
            
            # 2. On prédit la course
            X_r = pd.DataFrame([[grid_input, t_id, d_id, season_to_simulate]], columns=cols_race)
            pred_pos = model_race.predict(X_r)[0]
            
            predictions.append({'Pilote': row['DriverName'], 'Score': pred_pos})
        
        # --- D. RÉSULTAT ---
        results = pd.DataFrame(predictions).sort_values('Score')
        winner_predicted = results.iloc[0]['Pilote']
        
        # Verdict
        if winner_predicted == vrai_vainqueur:
            print(f" -> ✅ BINGO ! ({vrai_vainqueur})")
            score_card.append(1)
        else:
            # Est-ce qu'il était au moins dans le Top 3 prédit ?
            top3 = results.head(3)['Pilote'].values
            if vrai_vainqueur in top3:
                print(f" -> ⚠️ Presque (Top 3). IA: {winner_predicted} | Réel: {vrai_vainqueur}")
            else:
                print(f" -> ❌ Raté. IA: {winner_predicted} | Réel: {vrai_vainqueur}")
            score_card.append(0)

    # 4. BILAN FINAL
    if score_card:
        accuracy = (sum(score_card) / len(score_card)) * 100
        print(f"\n🏆 BILAN SAISON {season_to_simulate} ({mode_str})")
        print(f"Courses prédites : {len(score_card)}")
        print(f"Vainqueurs exacts : {sum(score_card)}")
        print(f"PRÉCISION DU MODÈLE : {accuracy:.1f}%")

if __name__ == "__main__":
    annee = int(input("Quelle saison voulez-vous simuler ? "))
    
    # Nouvel input pour le mode de grille
    choix_mode = input("Utiliser la VRAIE grille de départ pour la course ? (o/n) : ")
    use_real = choix_mode.lower() == 'o'
    
    run_simulation(annee, use_real_grid=use_real)