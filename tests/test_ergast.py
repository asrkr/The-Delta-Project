from the_delta_project.data.ergast import _fetch_race_result


def test_fetch_data_result_real_api():
    url = "https://api.jolpi.ca/ergast/f1/2024/results.json"
    df = _fetch_race_result(url)

    assert df is not None
    assert not isinstance(df, str)
    assert "DriverKey" in df.columns
    assert "TeamKey" in df.columns
    assert "circuitId" in df.columns
    assert df.iloc[0]["circuitId"] == "bahrain"
