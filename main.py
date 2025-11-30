from src.data_manager import load_data, update_database, update_calendar, get_rounds_for_race
from src.analysis import analyze_circuit_history

df = load_data()
if df is not None:
    ANNEE_CIBLE = 2025
    SEARCH_TERM = "Abu Dhabi"
    print(f"Recherche pour : {SEARCH_TERM}...")
    rounds_map, nom_officiel = get_rounds_for_race(SEARCH_TERM)
    if rounds_map:
        print(f"  -> Trouvé : {nom_officiel}")
        poids_qualif = analyze_circuit_history(df, rounds_map, nom_officiel, ANNEE_CIBLE)
