import pandas as pd
from src.data_manager import load_data
# AJOUT : On importe add_dual_form ici
from src.ml_model import encode_data, train_models, predict_race_outcome, add_dual_form

def run_simulation(season_to_simulate, use_real_grid=False):
    mode = "VRAIE GRILLE" if use_real_grid else "GRILLE PRÉDITE"
    print(f"\n--- SIMULATION SAISON {season_to_simulate} ({mode}) ---")
    
    df = load_data()
    if df is None: return

    # --- CORRECTION ICI ---
    # On doit d'abord calculer les formes (points/positions récentes)
    # avant de nettoyer les données, sinon l'IA ne trouve pas ses colonnes !
    df = add_dual_form(df)

    # Ensuite on encode
    df_clean, le_driver, le_team = encode_data(df)

    races = df[df['year'] == season_to_simulate].sort_values('round')['round'].unique()
    
    # initialisation des stats
    stats = {
        'total': 0, 'exact_win': 0, 'mae': [], 
        'top3': [], 'top5': [], 'top10': []
    }

    for race_round in races:
        print(f"\nRound {race_round}, vainqueur prédit : ", end="")
        
        # entraînement sur le passé
        mask_train = (df_clean['year'] < season_to_simulate) | \
                     ((df_clean['year'] == season_to_simulate) & (df_clean['round'] < race_round))
        train_data = df_clean[mask_train]
        
        if len(train_data) < 200: 
            print(".", end="")
            continue

        # on entraîne les modèles via la fonction partagée
        models = train_models(train_data)

        # réalité du jour
        current_race = df_clean[(df_clean['year'] == season_to_simulate) & (df_clean['round'] == race_round)]
        if current_race.empty: continue

        # prédiction via la fonction partagée
        # ATTENTION : On passe 'df' (le complet avec historique) pour récupérer les formes
        results = predict_race_outcome(models, current_race, season_to_simulate, race_round, le_driver, le_team, df, use_real_grid)
        
        # analyse des résultats
        results = results.merge(current_race[['DriverName', 'position']], left_on='Pilote', right_on='DriverName')
        results = results.sort_values('Course_Score')
        results['Pred_Pos'] = range(1, len(results) + 1)
        
        # calcul MAE
        mae = abs(results['Pred_Pos'] - results['position']).mean()
        stats['mae'].append(mae)
        
        # vainqueur
        winner_pred = results.iloc[0]['Pilote']
        winner_real = current_race.sort_values('position').iloc[0]['DriverName']
        
        if winner_pred == winner_real:
            stats['exact_win'] += 1
            print("✅", end="")
        else:
            print("❌", end="")

        # calcul de précision stricte (ordre exact)
        def get_strict_acc(n):
            top_ia = results.head(n)['Pilote'].tolist()
            top_real = current_race.sort_values('position').head(n)['DriverName'].tolist()
            matches = sum([1 for i in range(min(len(top_ia), len(top_real))) if top_ia[i] == top_real[i]])
            return (matches / n) * 100

        stats['top3'].append(get_strict_acc(3))
        stats['top5'].append(get_strict_acc(5))
        stats['top10'].append(get_strict_acc(10))
        stats['total'] += 1

    # bilan final
    if stats['total'] > 0:
        nb = stats['total']
        print(f"\n\n📊 BILAN {season_to_simulate}")
        print(f"Vainqueur : {(stats['exact_win']/nb)*100:.1f}%")
        print(f"Top 3 (Strict) : {sum(stats['top3'])/nb:.1f}%")
        print(f"Top 5 (Strict) : {sum(stats['top5'])/nb:.1f}%")
        print(f"Top 10 (Strict) : {sum(stats['top10'])/nb:.1f}%")
        print(f"MAE : {sum(stats['mae'])/nb:.2f}")

if __name__ == "__main__":
    annee = int(input("Saison : "))
    choix = input("Vraie grille ? (o/n) : ")
    run_simulation(annee, use_real_grid=(choix.lower() == 'o'))
