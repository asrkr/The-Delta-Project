from src.data_manager import load_data, get_rounds_for_race
from src.ml_model import train_and_predict

def main():
    df = load_data()

    if df is None:
        print("❌ Impossible de charger la base de données f1_data_complete.csv")
        return

    try:
        # --- INPUTS UTILISATEUR ---
        nom_gp = input("Quel Grand Prix voulez-vous prédire ? ")
        annee_cible = int(input("Sur quelle saison ? "))

        choix_grille = input("Voulez-vous utiliser la grille de départ réelle (si disponible) ? (o/n) : ")
        use_real = choix_grille.strip().lower() == "o"

        # --- IDENTIFICATION DE LA COURSE ---
        rounds_map, official_name = get_rounds_for_race(nom_gp)

        if not rounds_map or annee_cible not in rounds_map:
            print(f"❌ Course '{nom_gp}' introuvable pour la saison {annee_cible}.")
            return

        round_cible = rounds_map[annee_cible]

        print(f"\n--- 🏁 Prédiction : {official_name} {annee_cible} (Round {round_cible}) ---")

        # --- LANCEMENT DU ML ---
        train_and_predict(df, annee_cible, round_cible, official_name, use_real_grid=use_real)

    except Exception as e:
        print(f"⚠️ Erreur inattendue : {e}")


if __name__ == "__main__":
    main()
