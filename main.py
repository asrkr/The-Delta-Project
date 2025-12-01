from src.data_manager import load_data, get_rounds_for_race
from src.ml_model import train_and_predict


# chargement des données
df = load_data()

if df is not None:
    # configuration
    try:
        NOM_GP = input("Quel Grand Prix voulez-vous prédire ? ")
        ANNEE_CIBLE = int(input("Sur quelle saison ? "))
        # nouvelle question
        choix_grille = input("Voulez-vous utiliser la grille de départ réelle (si disponible) ? (o/n) : ")
        use_real = choix_grille.lower() == "o"

        rounds_map, official_name = get_rounds_for_race(NOM_GP) 
        print(f"--- Prédiction de l'algorithme : {NOM_GP} {ANNEE_CIBLE} ---")

        if rounds_map and ANNEE_CIBLE in rounds_map:
            round_cible = rounds_map[ANNEE_CIBLE]
            
            # on lance le machine learning avec nos options
            train_and_predict(df, ANNEE_CIBLE, round_cible, official_name, use_real_grid=use_real)

        else:
            print(f"Course non trouvée dans le calendrier {ANNEE_CIBLE}")

    except ValueError:
        print("Erreur : l'année doit être un nombre.")
