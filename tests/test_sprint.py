from the_delta_project.data.sprint import fetch_season_sprints


def test_fetch_season_sprints_before_2021() -> None:
    """Verify that seasons prior to 2021 return an empty DataFrame."""
    df = fetch_season_sprints(2020)
    assert df is not None
    assert df.empty


def test_fetch_season_sprints_api() -> None:
    """Test fetching season sprint results from Jolpica API."""
    df = fetch_season_sprints(2023)

    assert df is not None
    assert not df.empty
    assert "DriverKey" in df.columns
    assert "sprint_pos" in df.columns
    assert "sprint_grid" in df.columns
    assert "sprint_points" in df.columns
