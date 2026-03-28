import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.models.qualif_ranker import QualifRankerLGBM

# ---------------------------------------------------------
# 1) Encoding data
# ---------------------------------------------------------

def encode_data(df: pd.DataFrame) -> (pd.DataFrame, LabelEncoder, LabelEncoder, LabelEncoder):
    df_clean = df[df["status"].str.contains("Finished|Lap|Lapped", regex=True, na=False)].copy()

    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()

    team_col = "TeamKey" if "TeamKey" in df.columns else "Team"

    all_drivers = df["DriverKey"].astype(str).unique()
    all_teams = df[team_col].astype(str).unique()

    if "circuitId" in df.columns:
        all_circuits = df["circuitId"].astype(str).unique()
    else:
        all_circuits = ["unknown"]
        df_clean["circuitId"] = "unknown"

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    le_circuit.fit(all_circuits)

    df_clean["driver_id"] = le_driver.transform(df_clean["DriverKey"].astype(str))
    df_clean["team_id"] = le_team.transform(df_clean[team_col].astype(str))
    df_clean["circuit_id"] = le_circuit.transform(df_clean["circuitId"].astype(str))

    return df_clean, le_driver, le_team, le_circuit

# ---------------------------------------------------------
# 2) Model training
# ---------------------------------------------------------

def train_models(df_train: pd.DataFrame) -> (QualifRankerLGBM, RandomForestRegressor):
    # RandomForest hyperparameters
    params_qualif = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        # best parameters with tuning
        "n_estimators": 83,
        "learning_rate": 0.010417146488237577,
        "num_leaves": 60,
        "max_depth": -1,
        "min_child_samples": 26,
        "subsample": 0.9962990060021659,
        "colsample_bytree": 0.8896856637603093,
        "reg_lambda": 2.679269781861703,
        "reg_alpha": 0.7714673192056071
    }
    params_race = {
        "n_estimators": 320,
        "max_depth": 13,
        "min_samples_split": 13,
        "min_samples_leaf": 5,
        "max_features": None,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    }

    # Qualifying features
    features_qualif = [
        "team_id", "driver_id", "year",
        "form_grid", "circuit_importance", "circuit_id",
        "career_grid_avg", "circuit_grid_skill"
    ]
    # Filtering to keep only existing features
    features_qualif = [f for f in features_qualif if f in df_train.columns]

    model_qualif = QualifRankerLGBM(params=params_qualif)
    model_qualif.fit(df_train, features_qualif, target_col="grid")

    # Race features
    features_race = [
        "grid",
        "form_race",
        "career_race_avg",
        "pace_rank_season",
        "team_id", "driver_id", "year",
        "circuit_importance", "circuit_id",
        "circuit_race_skill",
        "career_race_pace", "career_clean_air_pace", "career_best_lap", "career_pit_loss", "career_wet_skill",
        "has_sprint", "sprint_delta",
        "is_rainy", "track_temp"
    ]
    features_race = [f for f in features_race if f in df_train.columns]

    model_race = RandomForestRegressor(**params_race)
    model_race.fit(df_train[features_race], df_train["position"])

    return model_qualif, model_race

# ---------------------------------------------------------
# TOOLS FOR BENCHMARKS
# ---------------------------------------------------------

def get_feature_importances(model, feature_names):
    """
    Return a sorted DataFrame of a sklearn model's feature importances
    """
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
