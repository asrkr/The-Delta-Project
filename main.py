from src.data_manager import load_data, get_rounds_for_race
from src.ml_model import train_and_predict

# chargement des données
df = load_data()

if df is not None:
    # Configuration
    NOM_GP = input("Quel Grand Prix voulez-vous prédire ? ")
    ANNEE_CIBLE = int(input("Sur quelle saison ? "))
    print(f"--- Prédiction de l'algorithme : {NOM_GP} {ANNEE_CIBLE} ---")
    rounds_map, official_name = get_rounds_for_race(NOM_GP)
    if rounds_map and ANNEE_CIBLE in rounds_map:
        round_cible = rounds_map[ANNEE_CIBLE]
        # on lance le machine learning
        train_and_predict(df, ANNEE_CIBLE, round_cible, official_name)
    else:
        print(f"Course non trouvée dans le calendrier 2025.")

