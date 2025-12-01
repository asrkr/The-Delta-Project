import pandas as pd
import requests
import time
import os


# on récupère les dossiers (actuel, celui des données, et des CSV)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
RESULTS_CSV_PATH = os.path.join(DATA_DIR, "f1_data_complete.csv")
CALENDAR_CSV_PATH = os.path.join(DATA_DIR, "races_calendar.csv")

# créer le dossier data si inexistant
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# fonction interne (anciennement get_race_data) - téléchargement des données avec insistance
def _fetch_race_result(url):
    max_retries = 4
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url, timeout=10)
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
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"Trop de requêtes (429). Pause de {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                print(f"Erreur serveur : {response.status_code}. Tentative {attempt+1}/{max_retries}")
                time.sleep(2)
                attempt += 1
        except Exception as e:
            print(f"Erreur technique : {e}")
    # échec complet après les essais
    print(f"Abandon définitif sur : {url}")
    return None


#  Partie 1 : télécharge les résultats, update le csv, et permet de filtrer les données qu'on veut par saison et course
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
            # pause pour le serveur
            time.sleep(1)
    if all_races:
        df_final = pd.concat(all_races, ignore_index=True)
        df_final.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"Base de données sauvegardée dans : {RESULTS_CSV_PATH} ({len(df_final)} lignes)")
    else:
        print("Aucune donnée récupérée.")


# Partie 2 : téléchargement du calendrier
def update_calendar(start_year=2019, end_year=2025):
    # on récupère la liste de toutes les courses
    print(f"Mise à jour du calendrier ({start_year}-{end_year})...")
    all_schedules = []
    for year in range(start_year, end_year+1):
        url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
        try:
            response = requests.get(url)
            data = response.json()
            races = data["MRData"]["RaceTable"]["Races"]
            for race in races:
                all_schedules.append({
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "raceName": race["raceName"],
                    "circuitId": race["Circuit"]["circuitId"],
                    "date": race["date"]
                })
            print(f"  -> Saison {year} : OK")
        except Exception as e:
            print(f"  Erreur {year} : {e}")
        # pause for the server
        time.sleep(1)
    df_calendar = pd.DataFrame(all_schedules)
    df_calendar.to_csv(CALENDAR_CSV_PATH, index=False)
    print(f"Calendrier enregistré dans : {CALENDAR_CSV_PATH}")


# Partie 3 : chargement des donnée
def load_data():
    # On charge le CSV s'il existe, sinon on propose de le télécharger
    if os.path.exists(RESULTS_CSV_PATH):
        return pd.read_csv(RESULTS_CSV_PATH)
    else:
        print("Fichier de donnée introuvable.")
        response = input("Voulez-vous le télécharger maintenant ? (o/n) ")
        if response.lower() == "o":
            update_database()
            time.sleep(5)  # pour laisser souffler le serveur entre les données et le calendrier
            return pd.read_csv(RESULTS_CSV_PATH)
        else:
            return None

# cherche dans le calendrier csv et renvoie un dictionnaire {année: round}
def get_rounds_for_race(race_name_keyword):
    if not os.path.exists(CALENDAR_CSV_PATH):
        print("Calendrier introuvable.\nTéléchargement...")
        update_calendar()
    df = pd.read_csv(CALENDAR_CSV_PATH)
    # on filtre avec notre paramètre d'entrée
    filtered = df[df["raceName"].str.contains(race_name_keyword, case=False, na=False)]
    if filtered.empty:
        print(f'Aucune course trouvée avec le nom "{race_name_keyword}".')
        return {}, None
    # on créé le dictionnaire final (nom du GP + calendrier dans la saison)
    official_name = filtered.iloc[0]["raceName"]
    rounds_map = dict(zip(filtered["year"], filtered["round"]))
    return rounds_map, official_name
