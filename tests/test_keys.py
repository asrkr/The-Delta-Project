import pandas as pd

from the_delta_project.utils.keys import canonicalize_driver_key, ensure_keys, team_lineage


def test_canonicalize_driver_key() -> None:
    """Verify driver key canonicalization and alias resolution."""
    assert canonicalize_driver_key("a_antonelli") == "k_antonelli"
    assert canonicalize_driver_key("leclerc") == "leclerc"
    assert canonicalize_driver_key("") == "unknown"


def test_team_lineage() -> None:
    """Verify team lineage mapping from Jolpica constructorId."""
    assert team_lineage("ferrari") == "ferrari"
    assert team_lineage("toro_rosso") == "rb_junior"
    assert team_lineage("alphatauri") == "rb_junior"
    assert team_lineage("renault") == "alpine"
    assert team_lineage("benetton") == "alpine"
    assert team_lineage("") == "unknown"


def test_ensure_keys() -> None:
    """Verify that DriverKey and TeamKey are guaranteed on input DataFrame."""
    df = pd.DataFrame(
        [
            {"DriverKey": "a_antonelli", "constructorId": "toro_rosso"},
            {"DriverKey": "leclerc", "constructorId": "ferrari"},
        ]
    )
    out = ensure_keys(df)
    assert list(out["DriverKey"]) == ["k_antonelli", "leclerc"]
    assert list(out["TeamKey"]) == ["rb_junior", "ferrari"]
