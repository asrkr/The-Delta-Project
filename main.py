import sys

from src.data_manager import load_data, get_rounds_for_race
from src.ml_model import train_and_predict

# Ensure emoji/UTF-8 output works on consoles with a legacy default encoding
# (e.g. Windows cp1252), which would otherwise raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    df = load_data()
 
    if df is None:
        print("❌ Unable to load database f1_data_complete.csv")
        return

    try:
        # --- USER INPUTS ---
        nom_gp = input("Which Grand Prix do you want to predict? ").strip()
        if not nom_gp:
            print("❌ Grand Prix name cannot be empty.")
            return

        try:
            annee_cible = int(input("Which season? "))
        except ValueError:
            print("❌ Invalid season. Please enter a valid year (e.g. 2025).")
            return

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
