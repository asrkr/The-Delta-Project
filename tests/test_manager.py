import os
import pandas as pd
from the_delta_project.data.manager import get_rounds_for_race, get_race_participants, update_calendar


def test_calendar_lookup():
    # Force updating calendar for 2024
    update_calendar(2024, 2024)
    rounds_map, official_name = get_rounds_for_race("Monaco")

    assert 2024 in rounds_map
    assert rounds_map[2024] == 8
    assert official_name == "Monaco Grand Prix"


def test_get_race_participants():
    # Mock dataframe
    mock_df = pd.DataFrame([
        {"year": 2024, "round": 1, "DriverKey": "c_leclerc", "DriverName": "Charles Leclerc", "Team": "Ferrari", "TeamKey": "ferrari", "grid": 2},
        {"year": 2024, "round": 1, "DriverKey": "m_verstappen", "DriveName": "Max Verstappen", "Team": "Red Bull", "TeamKey": "red_bull", "grid": 1},
    ])

    participants = get_race_participants(mock_df, 2024, 1)
    assert len(participants) == 2
    assert "DriverKey" in participants.columns
    assert "TeamKey" in participants.columns
