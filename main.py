from src.data_manager import load_data, get_rounds_for_race
from src.ml_model import train_and_predict

def main():
    df = load_data()
 
    if df is None:
        print("❌ Unable to load database f1_data_complete.csv")
        return

    try:
        # --- USER INPUTS ---
        nom_gp = input("Which Grand Prix do you want to predict? ")
        annee_cible = int(input("Which season? "))

        choix_grille = input("Do you want to use the real starting grid (if available)? (y/n): ")
        use_real = choix_grille.strip().lower() == "y"

        # --- RACE IDENTIFICATION ---
        rounds_map, official_name = get_rounds_for_race(nom_gp)

        if not rounds_map or annee_cible not in rounds_map:
            print(f"❌ Race '{nom_gp}' not found for season {annee_cible}.")
            return

        round_cible = rounds_map[annee_cible]

        print(f"\n--- 🏁 Prediction: {official_name} {annee_cible} (Round {round_cible}) ---")

        # --- START ML ---
        train_and_predict(df, annee_cible, round_cible, official_name, use_real_grid=use_real)

    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")


if __name__ == "__main__":
    main()
