import re
import unicodedata
import pandas as pd


# Alias mapping for driver keys across different data sources
DRIVER_KEY_ALIASES = {
    "a_antonelli": "k_antonelli",
}

TEAM_LINEAGE = {
    # Benetton -> Renault -> (Lotus F1) -> Renault -> Alpine
    "benetton": "alpine",
    "renault": "alpine",
    "lotus_f1": "alpine",

    # Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
    "jordan": "aston_martin",
    "midland": "aston_martin",
    "spyker": "aston_martin",
    "spyker_mf1": "aston_martin",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",

    # BAR -> Honda -> Brawn -> Mercedes
    "bar": "mercedes",
    "honda": "mercedes",
    "brawn": "mercedes",

    # Sauber family (BMW Sauber / Alfa Romeo / Kick Sauber / Audi)
    "bmw_sauber": "sauber",
    "alfa": "sauber",
    "audi": "sauber",

    # Red Bull family
    "jaguar": "red_bull",

    # RB junior (Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls)
    "minardi": "rb_junior",
    "toro_rosso": "rb_junior",
    "alphatauri": "rb_junior",
    "rb": "rb_junior",

    # Virgin -> Marussia -> Manor
    "virgin": "manor",
    "marussia": "manor",

    # Lotus Racing -> Caterham
    "lotus_racing": "caterham",

    # Misc normalization for modern naming variants
    "haas_f1_team": "haas",
    "haas": "haas",
}


def canonicalize_driver_key(raw_key: str) -> str:
    """Harmonizes driver aliases"""
    if raw_key is None or pd.isna(raw_key):
        return "unknown"
    s = str(raw_key).strip().lower()
    return DRIVER_KEY_ALIASES.get(s, s)


def team_lineage(constructor_id: str) -> str:
    """Returns team canonical key according to constructorId"""
    if constructor_id is None or pd.isna(constructor_id):
        return "unknown"
    cid = str(constructor_id).strip().lower()
    return TEAM_LINEAGE.get(cid, cid)


def ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Applies and ensures DriverKey and TeamKey presence for any DataFrame."""
    out = df.copy()
    if "DriverKey" in out.columns:
        out["DriverKey"] = out["DriverKey"].apply(canonicalize_driver_key)
    
    if "constructorId" in out.columns:
        out["TeamKey"] = out["constructorId"].apply(team_lineage)
    elif "TeamKey" in out.columns:
        out["TeamKey"] = out["TeamKey"].apply(team_lineage)
    
    return out
