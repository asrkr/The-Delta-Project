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
        try:
            plt.savefig(save_path)
            plt.close()
            print(f"  -> Graphique sauvegardé : images/{filename}")
        except Exception as e:
            print(f"  -> Erreur sauvegarde image : {e}")
        return corr
    else:
        print("Pas suffisemment de données fiables.")
        return 0


# analyse de la forme d'un pilote sur la saison de la course qu'on veut prédire
def analyze_driver_form(df, target_year, target_round):
    # si on veut analyser la première saison, alors on utilise la saison précédente complète comme indicateur de forme
    if target_round == 1:
        annee_ref = target_year - 1
        df_season = df[df["year"] == annee_ref].copy()
    # on sélectionne les donnée de la saison qu'on veut et on prend toutes les courses avant celle voulue
    else:
        annee_ref = target_year
        df_season = df[
            (df["year"] == target_year) &
            (df["round"] < target_round)
        ].copy()
    if df_season.empty:
        print(f"Aucune donnée trouvée pour la référence {annee_ref}")
        return None
    # nettoyage des données
    df_clean = df_season[
        df_season["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)
    ]
    # calcul des moyennes
    stats = df_clean.groupby("DriverName")[["grid", "position"]].mean()
    stats["count"] = df_clean.groupby("DriverName")["position"].count()

    # filtrage : si on utilise l'année d'avant (course 1), on veut des pilotes qui on fait minimum 5 courses
    # sinon, on est plus tolérant
    if target_round == 1:
        min_races = 5
    else:
        min_races = max(2, target_round * 0.2)
    stats = stats[stats["count"] >= min_races]
    # calcul de la différence entre départ et arrivée pour chaque pilote
    stats["delta"] = stats["grid"] - stats["position"]
    stats = stats.sort_values(by="position", ascending=True)
    return stats


# analyse de la carrière complète d'un pilote avec les données disponibles
def analyze_driver_career(df, target_year):
    # on va calculer la perf globale des pilotes depuis 2019 jusqu'à l'année cible (donc l'année avant la saison visée)
    df_career = df[df["year"] < target_year].copy()
    if df_career.empty:
        return None
    # nettoyage
    df_clean = df_career[
        df_career["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)
    ]
    # calculs
    stats = df_clean.groupby("DriverName")[["position"]].mean()
    stats.rename(columns={"positions": "career_positions"}, inplace=True)
    return stats
