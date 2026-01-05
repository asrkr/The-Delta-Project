import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

class QualifRankerLGBM(BaseEstimator, RegressorMixin):
    def __init__(self, params=None):
        # ✅ "eval_at" au lieu de "ndcg_eval_at" -> plus de warning
        self.params = params if params else {
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
        self.model = None
        self.feature_names = None

    def _prepare_data(self, df, features, target_col=None):
        df_sorted = df.sort_values(by=["year", "round"]).copy()

        # ---- Safety : types ----
        df_sorted["year"] = pd.to_numeric(df_sorted["year"], errors="coerce")
        df_sorted["round"] = pd.to_numeric(df_sorted["round"], errors="coerce")

        # LightGBM Ranker veut un grouping clair
        df_sorted = df_sorted.dropna(subset=["year", "round"])
        df_sorted["year"] = df_sorted["year"].astype(int)
        df_sorted["round"] = df_sorted["round"].astype(int)

        # ---- Target cleaning ----
        y = None
        if target_col is not None and target_col in df_sorted.columns:
            df_sorted[target_col] = pd.to_numeric(df_sorted[target_col], errors="coerce")

            # ✅ IMPORTANT : on enlève les grids invalides
            # grid <= 0 => pas une position qualif valide
            df_sorted = df_sorted[df_sorted[target_col].notna() & (df_sorted[target_col] > 0)].copy()

            # ✅ relevance par course (anti -1 garanti)
            group_size = df_sorted.groupby(["year", "round"], sort=False)[target_col].transform("size")
            y = (group_size + 1 - df_sorted[target_col]).astype(float)

            # double safety
            y = y.fillna(0.0).clip(lower=0.0)

        # ---- Features ----
        # cast catégories (LGBM accepte category dtype)
        cat_cols = ["team_id", "driver_id", "circuit_id"]
        for c in cat_cols:
            if c in df_sorted.columns:
                # NOTE: en training normalement pas de -1.
                # si jamais tu en as, remplace-les ici :
                df_sorted[c] = pd.to_numeric(df_sorted[c], errors="coerce").fillna(0).astype(int)
                df_sorted[c] = df_sorted[c].astype("category")

        X = df_sorted[features].copy()
        self.feature_names = features

        # ✅ group doit être calculé APRÈS filtrage
        group = df_sorted.groupby(["year", "round"], sort=False).size().to_list()

        return X, y, group, df_sorted

    def fit(self, df_train, features, target_col="grid"):
        X_train, y_train, group_train, _ = self._prepare_data(df_train, features, target_col)

        if y_train is None or len(y_train) == 0:
            raise ValueError("Training labels (y_train) are empty after cleaning. Check your grid data.")

        if sum(group_train) != len(X_train):
            raise ValueError("Group sizes do not match X_train length. (Data filtering/grouping mismatch)")

        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_at=[1, 3, 5],
            group=group_train,
            categorical_feature="auto",
        )
        return self

    def predict(self, df_test, features):
        X_test, _, group_test, df_sorted = self._prepare_data(df_test, features, target_col=None)

        if self.model is None:
            raise RuntimeError("The model is not trained.")

        raw_scores = self.model.predict(X_test)
        df_sorted["predicted_raw_score"] = raw_scores

        # rank 1 = best score
        df_sorted["predicted_rank"] = (
            df_sorted.groupby(["year", "round"], sort=False)["predicted_raw_score"]
            .rank(method="first", ascending=False)
        )

        # return aligned to df_test index
        result_map = df_sorted["predicted_rank"].to_dict()
        final_ranks = df_test.index.map(result_map).fillna(20).astype(int)
        return final_ranks
