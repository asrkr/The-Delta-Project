from the_delta_project.utils.keys import make_driver_key, make_team_key


def test_make_driver_key():
    assert make_driver_key("Charles", "Leclerc") == "c_leclerc"
    assert make_driver_key("Sergio", "Pérez") == "s_perez"
    assert make_driver_key("Andrea Kimi", "Antonelli") == "k_antonelli"
    assert make_driver_key("", "Leclerc") == "unknown"


def test_make_team_key():
    assert make_team_key("Ferrari") == "ferrari"
    assert make_team_key("Toro Rosso") == "rb_junior"
    assert make_team_key("AlphaTauri") == "rb_junior"
    assert make_team_key(" Red Bull Racing") == "red_bull"
    assert make_team_key("") == "unknown"
