import os
import pandas as pd

from the_delta_project.data.ergast import fetch_race_result

# Absolute paths based on project root
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dat"))
RACE_RESULTS_CSV_PATH = os.path.join(DATA_DIR, "race_results.csv")


def update_database(start_year: int = 2001, end_year: int = 2026) -> None:
    """Download race results from Ergast/Jolpica and save them incrementally to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_races = []

    for year in range(start_year, end_year+1):
        cpt = 0
        for rnd in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/results.json"
            result = fetch_race_result(url)

            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame) and not result.empty:
                result["year"] = int(year)
                result["round"] = int(rnd)
                all_races.append(result)
                cpt += 1

    if not all_races:
        return

    df_new = pd.concat(all_races, ignore_index=True)

    if os.path.exists(RACE_RESULTS_CSV_PATH):
        df_old = pd.read_csv(RACE_RESULTS_CSV_PATH)
        # Delete rewritten years to avoid duplicates
        df_old = df_old[(df_old["year"] < start_year) | (df_old["year"] > end_year)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values(["year", "round", "grid"])
    df_final.to_csv(RACE_RESULTS_CSV_PATH, index=False)
