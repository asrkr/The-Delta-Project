import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# charger le fichier csv
try:
    df = pd.read_csv("Data collection/f1_data_complete.csv")
    print(f"Chargement réussi. {len(df)} lignes importées.")
except FileNotFoundError:
    print("Erreur : le fichier CSV n'existe pas.")
    exit()

# on définit le numéro des courses correspondant à Abu Dhabi
abu_dhabi_map = {
    2019: 21,
    2020: 17,
    2021: 22,
    2022: 22,
    2023: 22,
    2024: 24
}
# on sélectionne que les courses de abu dhabi
df_abu_dhabi = pd.DataFrame()
for annee, course_num in abu_dhabi_map.items():
    course = df[(df["year"] == annee) & (df["round"] == course_num)]
    df_abu_dhabi = pd.concat([df_abu_dhabi, course])

# on enlève les pilotes qui ont DNF, donc à cause de crahs, pannes etc
# il ne restera que les pilotes qui ont fini ou qui se sont fait dépassé
df_clean = df_abu_dhabi[
    df_abu_dhabi["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)
]

print(f"Nombre de DNFs exclus : {len(df_abu_dhabi) - len(df_clean)}")

# visualisation et analyse
plt.figure(figsize=(12, 7))

# nuage de points
sns.scatterplot(
    data=df_clean,
    x="grid",
    y="position",
    hue="year",  # une couleur par année
    palette="viridis",
    s=100,   # taille des points
    alpha=0.8  # transparence
)

# ligne de référence de performance (y=x)
# si un point est sur la ligne, alors un pilote a fini à la même position où il a commencé
plt.plot([1, 20], [1, 20], "r--", linewidth=2, label="Position maintenue (Départ = Arrivée)")
plt.title("Abu Dhabi (2019 - 2024) : Impact de la qualif sur la course", fontsize=14)
plt.xlabel("Position de départ (grille)", fontsize=12)
plt.ylabel("Position d'arrivée", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
# inverser l'axe y pour avoir le premier en haut
plt.gca().invert_yaxis()
plt.savefig("abu_dhabi.png")

# calculer et afficher la corrélation
corr = df_clean["grid"].corr(df_clean["position"])
print("\n--- Résultats de l'analyse ---")
print(f"Corrélation Grille/Arrivée : {corr:.4f}")
if corr > 0.8:
    print('ANALYSE : circuit "procession", la qualif fait 80% du travail')
elif corr > 0.6:
    print("ANALYSE : Circuit Équilibré. La qualif est importante mais on peut remonter.")
else:
    print("ANALYSE : Circuit Chaotique. La grille ne veut rien dire (nombreux dépassements).")
