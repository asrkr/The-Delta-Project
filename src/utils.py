"""Shared anti-leakage utilities used by training, backtesting, and tuning scripts."""
import pandas as pd
import numpy as np


def make_train_mask(df: pd.DataFrame, season: int, race_round: int) -> pd.Series:
    """Return a boolean mask for rows strictly before (season, race_round)."""
    return (df["year"] < season) | ((df["year"] == season) & (df["round"] < race_round))


def circuit_impact_no_leak(df_window: pd.DataFrame, df_train_only: pd.DataFrame) -> pd.DataFrame:
    """
    Compute circuit_importance (grid/position correlation) from df_train_only rows only,
    then apply the resulting map to the full df_window.
    Rows without enough history default to 0.5.
    """
    df = df_window.copy()
    df["circuit_importance"] = 0.5

    needed = {"circuitId", "grid", "position"}
    if not needed.issubset(df.columns) or df_train_only.empty:
        return df

    train_finishers = df_train_only.dropna(subset=["circuitId", "grid", "position"]).copy()
    train_finishers = train_finishers[train_finishers["position"] > 0]

    if train_finishers.empty:
        return df

    def corr_grid_pos(g: pd.DataFrame) -> float:
        if g["grid"].nunique() < 2 or g["position"].nunique() < 2:
            return 0.5
        c = g["grid"].corr(g["position"])
        return float(c) if pd.notna(c) else 0.5

    impact_map = (
        train_finishers
        .groupby("circuitId", dropna=True)[["grid", "position"]]
        .apply(corr_grid_pos)
        .to_dict()
    )

    df["circuit_importance"] = df["circuitId"].map(impact_map).fillna(0.5)
    return df
