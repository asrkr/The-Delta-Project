import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# On gère les chemins pour les images
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
IMG_DIR = os.path.join(PROJECT_ROOT, "images")

# si le dossier images n'existe pas, on le créé
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)


# analyse l'historique d'une course, jusqu'à l'année cible (exclue)
def analyze_circuit_history(df, circuit_round_map, gp_name, target_year):
    # 1: on filtre et on garde que les années AVANT l'année qu'on veut prédire
    df_history = df[df["year"] < target_year].copy()
    # on séléctionne les données spécifique du circuit voulu
    df_circuit = pd.DataFrame()
    for year, round_num in circuit_round_map.items():
        # on vérifie que l'année soit bien passée
        if year < target_year:
            course = df_history[(df_history["year"] == year) & (df_history["round"] == round_num)]
            df_circuit = pd.concat([df_circuit, course])
        if df_circuit.empty:
            print("  Aucune donnée historique trouvée pour ce circuit avant cette date.")
            return 0
        # nettoyage : on enlève les DNF
        df_clean = df_circuit[
            df_circuit["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)
        ]
        # calcul de la corrélation
        if len(df_clean) > 10:  # il faut un minimum de données pour que ça fasse sens
            corr = df_clean["grid"].corr(df_clean["position"])
            print(f"  -> Corrélation Qualif/Course : {corr:.4f}")
            # génération du graphique
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df_clean,
                x="grid",
                y="position",
                hue="year",
                palette="viridis",
                s=80
            )
            # ligne de référence pour le graphique
            plt.plot([1, 20], [1, 20], "r--", label="Maintien position")
            # itre et légendes
            plt.title(f"{gp_name} : historique (pré-{target_year} - Corrélation : {corr:.4f})")
            plt.xlabel("Position au départ")
            plt.ylabel("Position à l'arrivée")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()  # 1er en haut
            # sauvegarde de l'image avec nom de fichier dynamique
            new_name = gp_name.replace(" ", "_")
            filename = f"analysis_{new_name}_{target_year}.png"
            save_path = os.path.join(IMG_DIR, filename)
            plt.close()
            print(f"  -> Graphique sauvegardé : images/{filename}")
            return corr

