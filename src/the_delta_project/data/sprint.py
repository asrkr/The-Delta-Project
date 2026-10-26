import pandas as pd

from the_delta_project.data.ergast import _entries_to_df, _fetch_races


def fetch_season_sprint(year: int) -> pd.DataFrame:
    """Fetch all sprint race results for a given season from Jolpica API.
    
    Sprint races were introduced in 2021. Returns an empty DataFrame if
    year < 2021 or if no sprint sessions were held during the season.
    """
    if year < 2021:
        return pd.DataFrame()
    
    df = _entries_to_df(_fetch_races(f"{year}/sprint", "SprintResults"), "SprintResults")
    if df.empty:
        return df
    
    return df.rename(
        columns = {
            "position": "sprint_pos",
            "grid": "sprint_grid",
            "points": "sprint_points",
        }
    )
