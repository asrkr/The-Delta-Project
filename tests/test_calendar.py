from the_delta_project.data.calendar import fetch_season_calendar


def test_fetch_season_calendar_api() -> None:
    """Test fetching full season race calendar from Jolpica API."""
    df = fetch_season_calendar(2024)

    assert df is not None
    assert not df.empty
    assert "year" in df.columns
    assert "round" in df.columns
    assert "raceName" in df.columns
    assert "circuitId" in df.columns

    # Verify Bahrain 2024 round 1
    round_1 = df[df["round"] == 1].iloc[0]
    assert round_1["circuitId"] == "bahrain"
    assert round_1["year"] == 2024
