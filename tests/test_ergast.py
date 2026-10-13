from the_delta_project.data.ergast import fetch_race_result, fetch_qualifying_results


def test_fetch_race_result_api():
    url = "https://api.jolpi.ca/ergast/f1/2024/results.json"
    df = fetch_race_result(url)

    assert df is not None
    assert not isinstance(df, str)
    assert "DriverKey" in df.columns
    assert "TeamKey" in df.columns
    assert "circuitId" in df.columns
    assert df.iloc[0]["circuitId"] == "bahrain"


def test_fetch_qualifying_results_api():
    df = fetch_qualifying_results(2024, 1)

    assert df is not None
    assert "DriverKey" in df.columns
    assert "TeamKey" in df.columns
    assert "grid" in df.columns
    assert df.iloc[0]["grid"] == 1
