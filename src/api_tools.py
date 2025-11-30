import requests
import pandas as pd
import time


def get_race_data(url):
    # On va tenter 3 fois maximum avant d'abandonner
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. La requête
            response = requests.get(url, timeout=10)  # On ajoute un timeout de 10s
            # Si le serveur renvoie une erreur (ex: 500 ou 404), on réessaie
            if response.status_code != 200:
                print(f"Erreur serveur (Code {response.status_code}). Tentative {attempt + 1}/{max_retries}...")
                time.sleep(2)   # On attend 2 secondes avant de relancer
                continue
            data = response.json()
            # 2. Vérification : Est-ce que la course existe ?
            # C'est le SEUL cas où on veut renvoyer None immédiatement (fin de saison valide)
            races_list = data["MRData"]["RaceTable"]["Races"]
            if not races_list:
                return "FIN_DE_SAISON"  # On renvoie un signal clair, pas juste None
            # 3. Si on est là, c'est qu'on a des données ! On traite.
            resultats = races_list[0]["Results"]
            df = pd.DataFrame(resultats)
            # Nettoyage habituel
            df["DriverName"] = df["Driver"].apply(lambda x: x["familyName"])
            df["Team"] = df["Constructor"].apply(lambda x: x["name"])
            df["grid"] = pd.to_numeric(df["grid"])
            df["position"] = pd.to_numeric(df["position"])
            colonnes_utiles = ["DriverName", "Team", "grid", "position", "points", "status"]
            return df[colonnes_utiles]
        except Exception as e:
            print(f"Bug connexion ({e}). Tentative {attempt + 1}/{max_retries}...")
            time.sleep(2)
            continue   # On retourne au début de la boucle for
    # Si on arrive ici, c'est qu'on a échoué 3 fois de suite
    print(f"Abandon sur l'URL : {url}")
    return None   # Vrai échec technique
