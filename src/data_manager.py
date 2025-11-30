import pandas as pd
import requests
import time
import os


# on récupère les dossiers (actuel, celui des données, et du CSV)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CSV_PATH = os.path.join(DATA_DIR, "f1_data_complete.csv")

# créer le dossier data si inexistant
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# fonction interne (anciennement get_race_data)
def _fetch_race_result(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                return "END_OF_SEASON"
            # extraction et nettoyage des données
            df = pd.DataFrame(races[0]["Results"])
            df["DriverName"] = df["Driver"].apply(lambda x: x["familyName"])
            df["Team"] = df["Constructor"].apply(lambda x: x["name"])
            df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
            df["position"] = pd.to_numeric(df["position"], errors="coerce")
            # on garde uniquement les colonnes nécessaires
            cols = ["DriverName", "Team", "grid", "position", "status"]
            return df[cols]
    except Exception as e:
        print(f"Erreur technique : {e}")
    return None


# fonction principale 1 : update le csv, et permet de filtrer les données qu'on veut par saison et course
def update_database(start_year=2019, end_year=2025):
    # télécharge les données et écrase le CSV
    print(f"Mise à jour de la base de donnée ({start_year}-{end_year})...")
    all_races = []
    for year in range(start_year, end_year + 1):
        print(f"Traitement de la saison {year}...")
        for round_num in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_num}/results.json"
            result = _fetch_race_result(url)
            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame):
                result["year"] = year
                result["round"] = round_num
                all_races.append(result)
            time.sleep(0.5)
    if all_races:
        df_final = pd.concat(all_races, ignore_index=True)
        df_final.to_csv(CSV_PATH, index=False)
        print(f"Base de données sauvegardée dans : {CSV_PATH}")
    else:
        print("Aucune donnée récupérée.")


# fonction principale 2 : chargement des données
def load_data():
    # On charge le CSV s'il existe, sinon on propose de le télécharger
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        print("Fichier de donnée introuvable.")
        response = input("Voulez-vous le télécharger maintenant ? (o/n)")
        if response.lower() == "o":
            update_database()
            return pd.read_csv(CSV_PATH)
        else:
            return None
