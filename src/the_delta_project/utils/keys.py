import re
import unicodedata


# Alias mapping for driver keys across different data sources
DRIVER_KEY_ALIASES = {
    "a_antonelli": "k_antonelli",
}

TEAM_KEY_ALIASES = {
    # Benetton -> Renault -> (Lotus F1) -> Renault -> Alpine
    "benetton": "alpine",
    "renault": "alpine",
    "alpine": "alpine",
    "alpine_f1_team": "alpine",
    "lotus_f1": "alpine",
    "lotus_f1_team": "alpine",

    # Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
    "jordan": "aston_martin",
    "midland": "aston_martin",
    "spyker": "aston_martin",
    "spyker_mf1": "aston_martin",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",

    # Sauber family (BMW Sauber / Alfa Romeo / Kick Sauber / Audi)
    "sauber": "sauber",
    "bmw_sauber": "sauber",
    "kick_sauber": "sauber",
    "alfa_romeo": "sauber",
    "alfa_romeo_racing": "sauber",
    "audi": "sauber",

    # Red Bull family
    "jaguar": "red_bull",
    "red_bull": "red_bull",
    "red_bull_racing": "red_bull",

    # RB junior (Minardi -> Toro Rosso -> AlphaTauri -> RB -> Racing Bulls)
    "minardi": "rb_junior",
    "toro_rosso": "rb_junior",
    "alphatauri": "rb_junior",
    "rb": "rb_junior",
    "rb_f1_team": "rb_junior",
    "racing_bulls": "rb_junior",

    # Misc normalization for modern naming variants
    "haas_f1_team": "haas",
    "haas": "haas",
}


def clean_text(text: str) -> str:
    """Base normalization: strips accents, lowercases, and handles missing /NaN values."""
    if not text or str(text).strip().lower() in ("nan", "none", "unknown", "null"):
        return ""
    normalized = unicodedata.normalize("NFKD", str(text).strip().lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def canonicalize_driver_key(raw_key: str) -> str:
    """Map a driver key to its canonical alias if present"""
    if not raw_key:
        return "unknown"
    key = str(raw_key).strip().lower()
    return DRIVER_KEY_ALIASES.get(key, key)


def make_driver_key(given_name: str, family_name: str) -> str:
    """Build a stable driver key (e.g., 'c_leclerc')."""
    raw_given_name = clean_text(given_name)
    raw_family_name = clean_text(family_name)
    # Clean non-alphabetic characters for drivers (e.g., 'Pérez' -> 'perez')
    clean_given_name = re.sub(r"[^a-z]", "", raw_given_name)
    clean_family_name = re.sub(r"[^a-z]", "", raw_family_name)

    if not clean_given_name or not clean_family_name:
        return "unknown"

    raw_key = f"{clean_given_name[0]}_{clean_family_name}"
    return canonicalize_driver_key(raw_key)


def make_team_key(team_name: str) -> str:
    """Build a stable team key (e.g. 'red_bull')."""
    raw_team = clean_text(team_name)
    # Convert spaces/symbols to underscores for teams (e.g., 'Toro Rosso' -> 'toro_rosso')
    clean_team = re.sub(r"[^a-z0-9]+", "_", raw_team).strip("_")
    if not clean_team:
        return "unknown"
    return TEAM_KEY_ALIASES.get(clean_team, clean_team)
