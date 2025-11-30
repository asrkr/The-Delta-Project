from src.api_tools import get_race_data
import pandas as pd
import time


all_races_data = []

print("--- Démarrage du téléchargement ---")

for annee in range(2019, 2026):
    print(f"Traitement de la saison {annee}...")
    for course_num in range(1, 26):
        url = f"https://api.jolpi.ca/ergast/f1/{annee}/{course_num}/results.json"
        # Appel de la fonction
        result = get_race_data(url)
        # on vérifie d'abord si le résultat est un tableau
        if isinstance(result, pd.DataFrame):
            result["year"] = annee
            result["round"] = course_num
            all_races_data.append(result)
            print(f"  -> Course {course_num} : OK")
        # si c'est pas un tableau, c'est le signal de fin ?
        elif isinstance(result, str) and result == "FIN_DE_SAISON":
            print(f"  -> Fin de la saison {annee} détectée au round {course_num - 1}")
            break
        else:
            print(f"  -> Course {course_num} sautée (erreur technique ou pas de réponse).")
        time.sleep(0.3)

if all_races_data:
    df_final = pd.concat(all_races_data, ignore_index=True)
    print("\n--- TERMINE ! ---")
    print(f"Lignes totales : {len(df_final)}")
    df_final.to_csv("f1_data_complete.csv", index=False)
    print("Sauvegarde effectuée.")
