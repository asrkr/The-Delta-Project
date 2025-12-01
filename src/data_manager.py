import pandas as pd
import requests
import time
import os

# on récupère les dossiers
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
RESULTS_CSV_PATH = os.path.join(DATA_DIR, "f1_data_complete.csv")
CALENDAR_CSV_PATH = os.path.join(DATA_DIR, "races_calendar.csv")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


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
                
                df = pd.DataFrame(races[0]["Results"])
                
                # Sécurisation avec .get pour éviter les erreurs sur les vieilles saisons
                if "Driver" in df.columns:
                    df["DriverName"] = df["Driver"].apply(lambda x: x.get("familyName", ""))
                if "Constructor" in df.columns:
                    df["Team"] = df["Constructor"].apply(lambda x: x.get("name", ""))
                
                df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
                df["position"] = pd.to_numeric(df["position"], errors="coerce")
                
                # Filtrage des colonnes existantes
                cols_ok = ["DriverName", "Team", "grid", "position", "status"]
                final_cols = [c for c in cols_ok if c in df.columns]
                return df[final_cols]

            elif response.status_code == 429:
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                time.sleep(2)
                attempt += 1
        except Exception:
            time.sleep(2)
            attempt += 1
    return None


def update_database(start_year=2001, end_year=2025):
    print(f"Mise à jour complète ({start_year}-{end_year})...")
    all_races = []
    for year in range(start_year, end_year + 1):
        print(f"Saison {year}...", end=" ")
        cpt = 0
        for round_num in range(1, 26):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_num}/results.json"
            result = _fetch_race_result(url)
            if isinstance(result, str) and result == "END_OF_SEASON":
                break
            elif isinstance(result, pd.DataFrame):
                result["year"] = year
                result["round"] = round_num
                all_races.append(result)
                cpt += 1
            time.sleep(0.8)
        print(f"({cpt} courses)")

    if all_races:
        df_final = pd.concat(all_races, ignore_index=True)
        df_final.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"Sauvegardé : {RESULTS_CSV_PATH}")
    else:
        print("Aucune donnée.")


# NOUVELLE FONCTION : Télécharge une seule saison et met à jour le fichier existant
def download_single_season(year):
    print(f"Téléchargement saison {year}...")
    new_races = []
    
    for round_num in range(1, 26):
        url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_num}/results.json"
        result = _fetch_race_result(url)
        
        if isinstance(result, str) and result == "END_OF_SEASON":
            break
        elif isinstance(result, pd.DataFrame):
            result["year"] = year
            result["round"] = round_num
            new_races.append(result)
            print(f"Round {round_num}: OK")
        time.sleep(0.8)

    if not new_races:
        print("Aucune donnée.")
        return

    df_new_season = pd.concat(new_races, ignore_index=True)
    
    # Fusion avec l'existant
    if os.path.exists(RESULTS_CSV_PATH):
        df_total = pd.read_csv(RESULTS_CSV_PATH)
        # On supprime les anciennes données de cette année pour éviter les doublons
        df_total = df_total[df_total["year"] != year]
        df_final = pd.concat([df_total, df_new_season], ignore_index=True)
    else:
        df_final = df_new_season

    df_final = df_final.sort_values(by=["year", "round"])
    df_final.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Mise à jour terminée pour {year}.")


def update_calendar(start_year=2001, end_year=2025):
    print(f"Mise à jour calendrier ({start_year}-{end_year})...")
    all_schedules = []
    for year in range(start_year, end_year+1):
        url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
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
        except Exception: 
            pass
        time.sleep(1)
    pd.DataFrame(all_schedules).to_csv(CALENDAR_CSV_PATH, index=False)
    print("Calendrier mis à jour.")


def load_data():
    if os.path.exists(RESULTS_CSV_PATH):
        return pd.read_csv(RESULTS_CSV_PATH)
    else:
        print("Fichier introuvable.")
        return None


def get_rounds_for_race(race_name_keyword):
    if not os.path.exists(CALENDAR_CSV_PATH):
        update_calendar()
    try:
        df = pd.read_csv(CALENDAR_CSV_PATH)
        filtered = df[df["raceName"].str.contains(race_name_keyword, case=False, na=False)]
        if filtered.empty: return {}, None
        return dict(zip(filtered["year"], filtered["round"])), filtered.iloc[0]["raceName"]
    except: return {}, None


# v1.4 : nouvelle fonction pour récupérer la grille des pilotes
def get_race_participants(df, target_year, target_round):
    # 1 : on cherche la course exacte
    participants = df[
        (df["year"] == target_year) &
        (df["round"] == target_round)
    ].sort_values("grid")

    if not participants.empty:
        # cas 1 : historique trouvé
        cols = ["DriverName", "Team"]
        if "grid" in participants.columns:
            cols.append("grid")
        return participants[cols].drop_duplicates()
    # 2 : cas futur, on cherche la dernière course disponible de la même année
    same_year_races = df[df["year"] == target_year]

    if not same_year_races.empty:
         # on prend le round max de cette année
         last_round = same_year_races["round"].max()
         fallback = same_year_races[same_year_races["round"] == last_round]
         return fallback[["DriverName", "Team"]].drop_duplicates()
    
    # 3 : cas de nouvelle saison : on prend la dernière course de l'année dernière
    prev_year_races = df[df["year"] == target_year - 1]

    if not prev_year_races.empty:
        last_round = prev_year_races["round"].max()
        fallback = prev_year_races[prev_year_races["round"] == last_round]
        return fallback[["DriverName", "Team"]].drop_duplicates()
    
    return pd.DataFrame()  # rien trouvé


if __name__ == "__main__":
    print("1. Télécharger une saison spécifique")
    print("2. Mise à jour complète (2001-2025)")
    print("3. Mettre à jour le calendrier")
    
    choix = input("Choix : ")
    
    if choix == "1":
        annee = int(input("Année : "))
        download_single_season(annee)
    elif choix == "2":
        update_database()
    elif choix == "3":
        update_calendar()
