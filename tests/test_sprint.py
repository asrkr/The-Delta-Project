from the_delta_project.data.sprint import fetch_sprint_results


def test_fetch_sprint_results_before_2021():
    assert fetch_sprint_results(2020, 1) is None


def test_fetch_sprint_results_api():
    # Brazil 2023 (round 20) had a sprint
    df = fetch_sprint_results(2023, 20)

    assert df is not None
    assert "DriverKey" in df.columns
    assert "sprint_pos" in df.columns
    assert "status" in df.columns
    assert df.iloc[0]["sprint_pos"] == 1
