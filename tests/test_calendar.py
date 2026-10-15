from the_delta_project.data.calendar import fetch_calendar


def test_fetch_calendar_api():
    df = fetch_calendar(2024, 2024)

    assert df is not None
    assert not df.empty
    assert "year" in df.columns
    assert "round" in df.columns
    assert "raceName" in df.columns
    assert "circuitId" in df.columns
    assert "date" in df.columns

    # Verify Bahrain 2024 round 1
    round_1 = df[df["round"] == 1].iloc[0]
    assert round_1["circuitId"] == "bahrain"
    assert round_1["year"] == 2024
