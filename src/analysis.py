import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# On gère les chemins pour les images
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(CURRENT_DIR, "images"))

# si le dossier images n'existe pas, on le créé
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)


# analyse l'historique d'une course, jusqu'à l'année cible (exclue)
def analyze_circuit_history(df, circuit_round_map, target_year):
    # 1: on filtre et on garde que les années AVANT l'année qu'on veut prédire
    df_history = df[df["year"] < target_year].copy()
    # on séléctionne les bonnes courses (dans ce cas, Abu Dhabi)
    
