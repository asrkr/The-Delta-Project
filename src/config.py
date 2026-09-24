"""Central configuration: single source of truth for model hyperparameters
and feature lists.

Previously these were duplicated between ``ml_model.train_models`` and
``QualifRankerLGBM.__init__`` (and the feature lists were inlined). Keeping them
here avoids drift between the model definition and the training code.
"""

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# LightGBM Ranker (qualifying "brain") — tuned via Optuna (see dev_tools/).
QUALIF_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    # best parameters found with tuning
    "n_estimators": 83,
    "learning_rate": 0.010417146488237577,
    "num_leaves": 60,
    "max_depth": -1,
    "min_child_samples": 26,
    "subsample": 0.9962990060021659,
    "colsample_bytree": 0.8896856637603093,
    "reg_lambda": 2.679269781861703,
    "reg_alpha": 0.7714673192056071,
}

# RandomForest (race "brain").
RACE_PARAMS = {
    "n_estimators": 320,
    "max_depth": 13,
    "min_samples_split": 13,
    "min_samples_leaf": 5,
    "max_features": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

QUALIF_FEATURES = [
    "team_id", "driver_id", "year",
    "form_grid", "circuit_importance", "circuit_id",
    "career_grid_avg", "circuit_grid_skill",
]

RACE_FEATURES = [
    "grid",
    "form_race",
    "career_race_avg",
    "pace_rank_season",
    "team_id", "driver_id", "year",
    "circuit_importance", "circuit_id",
    "circuit_race_skill",
    "career_race_pace", "career_clean_air_pace", "career_best_lap",
    "career_pit_loss", "career_wet_skill",
    "has_sprint", "sprint_delta",
    "is_rainy", "track_temp",
]
