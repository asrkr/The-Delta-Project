from the_delta_project.data.ergast import fetch_qualifying_results, fetch_season_results


def test_fetch_season_results() -> None:
    """Test fetching full season race results from Jolpica API."""
    df = fetch_season_results(2024)

    assert df is not None
    assert not df.empty
    assert "DriverKey" in df.columns
    assert "TeamKey" in df.columns
    assert "circuitId" in df.columns

    round_1 = df[df["round"] == 1]
    assert not round_1.empty
    assert round_1.iloc[0]["circuitId"] == "bahrain"


def test_fetch_qualifying_results_api() -> None:
    """Test fetching single round qualifying result from Jolpica API."""
    df = fetch_qualifying_results(2024, 1)

    assert df is not None
    assert not df.empty
    assert "DriverKey" in df.columns
    assert "TeamKey" in df.columns
    assert "grid" in df.columns
    assert df.iloc[0]["grid"] == 1.0
