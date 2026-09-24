"""The Delta Project — F1-themed Streamlit dashboard.

Three views:
  🔮 Prédiction   — predict a single Grand Prix (Oracle or Analyst mode)
  🔄 Données      — refresh the datasets (Ergast / FastF1 / sprints / calendar)
  🛠️ Mode Dev     — full-season walk-forward backtest (simulateur_saison)

Run with:  streamlit run app.py
"""
import io
import os
import sys
import contextlib
from datetime import datetime

import pandas as pd
import streamlit as st

# Make sure the project root is importable (src/, dev_tools/) regardless of CWD.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_manager import (
    load_data,
    get_rounds_for_race,
    update_database,
    update_calendar,
    update_sprint_data,
    update_latest_qualifying,
    extract_fastf1_features,
    DATA_DIR,
    RESULTS_CSV_PATH,
    CALENDAR_CSV_PATH,
    EXTRA_CSV_PATH,
    SPRINT_CSV_PATH,
    QUALI_CSV_PATH,
)
from src.ml_model import train_and_predict

# The season simulator lives in dev_tools/ (not part of the packaged src/).
try:
    from dev_tools.simulateur_saison import run_simulation
    HAS_SIMULATOR = True
except Exception:
    run_simulation = None
    HAS_SIMULATOR = False


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

F1_RED = "#E10600"

# Approximate 2024/2025 team colours (matched loosely by name substring).
TEAM_COLORS = {
    "red bull": "#3671C6",
    "ferrari": "#E8002D",
    "mercedes": "#27F4D2",
    "mclaren": "#FF8000",
    "aston martin": "#229971",
    "alpine": "#0093CC",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "racing bulls": "#6692FF",
    "sauber": "#52E252",
    "kick sauber": "#52E252",
    "alfa romeo": "#900000",
    "haas": "#B6BABD",
}


def team_color(team: str) -> str:
    t = str(team).lower()
    for key, color in TEAM_COLORS.items():
        if key in t:
            return color
    return "#888888"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Titillium Web', sans-serif;
        }}
        .stApp {{
            background: radial-gradient(circle at 20% 0%, #1d1d2b 0%, #15151e 55%, #0e0e15 100%);
        }}
        /* Header banner */
        .delta-header {{
            display: flex; align-items: center; gap: 18px;
            padding: 22px 26px; border-radius: 14px;
            background: linear-gradient(100deg, #1f1f2b 0%, #26111a 100%);
            border: 1px solid #2e2e3c;
            border-left: 6px solid {F1_RED};
            margin-bottom: 6px;
        }}
        .delta-title {{
            font-size: 2.1rem; font-weight: 900; letter-spacing: 1px;
            color: #fff; margin: 0; line-height: 1;
        }}
        .delta-title .accent {{ color: {F1_RED}; }}
        .delta-sub {{ color: #9a9aa8; font-size: .95rem; margin-top: 4px; }}
        /* Checkered strip */
        .checker {{
            height: 10px; border-radius: 4px; margin: 8px 0 22px 0;
            background-image:
                linear-gradient(45deg, #2c2c38 25%, transparent 25%),
                linear-gradient(-45deg, #2c2c38 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #2c2c38 75%),
                linear-gradient(-45deg, transparent 75%, #2c2c38 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            border: 1px solid #2c2c38;
        }}
        /* Podium cards */
        .podium-wrap {{ display: flex; gap: 14px; margin: 6px 0 22px 0; }}
        .podium {{
            flex: 1; border-radius: 12px; padding: 16px 18px;
            background: #1c1c27; border: 1px solid #2e2e3c;
            border-top: 4px solid #555;
        }}
        .podium.p1 {{ border-top-color: #FFD700; }}
        .podium.p2 {{ border-top-color: #C0C0C0; }}
        .podium.p3 {{ border-top-color: #CD7F32; }}
        .podium .pos {{ font-size: .8rem; color: #9a9aa8; font-weight: 700; letter-spacing: 1px; }}
        .podium .name {{ font-size: 1.25rem; font-weight: 700; color: #fff; margin: 2px 0; }}
        .podium .team {{ font-size: .85rem; }}
        .podium .delta {{ font-size: .8rem; margin-top: 6px; }}
        /* Results table */
        table.delta-table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
        table.delta-table th {{
            text-align: left; color: #9a9aa8; font-weight: 600;
            border-bottom: 2px solid {F1_RED}; padding: 8px 10px; font-size: .8rem;
            text-transform: uppercase; letter-spacing: .5px;
        }}
        table.delta-table td {{ padding: 8px 10px; border-bottom: 1px solid #24242f; color: #eee; }}
        table.delta-table tr:hover td {{ background: #1d1d28; }}
        .pos-badge {{
            display: inline-block; width: 26px; height: 26px; line-height: 26px;
            text-align: center; border-radius: 6px; font-weight: 700;
            background: #2a2a36; color: #fff;
        }}
        .pos-badge.p1 {{ background: #FFD700; color: #15151e; }}
        .pos-badge.p2 {{ background: #C0C0C0; color: #15151e; }}
        .pos-badge.p3 {{ background: #CD7F32; color: #15151e; }}
        .team-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
        .up {{ color: #38d66b; font-weight: 700; }}
        .down {{ color: #ff5252; font-weight: 700; }}
        .flat {{ color: #888; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown(
        """
        <div class="delta-header">
            <div style="font-size:2.4rem;">🏎️</div>
            <div>
                <p class="delta-title">THE <span class="accent">DELTA</span> PROJECT</p>
                <div class="delta-sub">F1 race prediction engine · Dual&nbsp;Brain (LightGBM&nbsp;+&nbsp;RandomForest)</div>
            </div>
        </div>
        <div class="checker"></div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_calendar() -> pd.DataFrame:
    if not os.path.exists(CALENDAR_CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(CALENDAR_CSV_PATH)


@st.cache_data(show_spinner="Chargement de la base de données…")
def load_dataset() -> pd.DataFrame:
    return load_data()


@contextlib.contextmanager
def capture_logs():
    """Capture stdout produced by the underlying pipeline (it uses print())."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def file_status_row(label: str, path: str) -> dict:
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        rows = ""
        try:
            rows = f"{len(pd.read_csv(path)):,}"
        except Exception:
            rows = "—"
        return {"Fichier": label, "État": "✅", "Lignes": rows, "Taille": f"{size_kb:,.0f} Ko", "Modifié": mtime}
    return {"Fichier": label, "État": "❌", "Lignes": "—", "Taille": "—", "Modifié": "absent"}


# ---------------------------------------------------------------------------
# Tab 1 — Prediction
# ---------------------------------------------------------------------------

def render_prediction_tab() -> None:
    cal = load_calendar()
    if cal.empty:
        st.warning("📅 Calendrier introuvable. Lance d'abord une mise à jour dans l'onglet **🔄 Données**.")
        return

    race_names = sorted(cal["raceName"].dropna().unique().tolist())
    default_idx = race_names.index("Monaco Grand Prix") if "Monaco Grand Prix" in race_names else 0

    c1, c2, c3 = st.columns([3, 1.4, 2])
    with c1:
        gp = st.selectbox("Grand Prix", race_names, index=default_idx)
    # Seasons available for this race
    rounds_map, official = get_rounds_for_race(gp)
    seasons = sorted(rounds_map.keys(), reverse=True) if rounds_map else []
    with c2:
        if seasons:
            season = st.selectbox("Saison", seasons, index=0)
        else:
            season = st.number_input("Saison", min_value=2001, max_value=2030, value=2025, step=1)
    with c3:
        mode = st.radio(
            "Mode",
            ["🔮 Oracle (grille prédite)", "🔬 Analyst (grille réelle)"],
            help="Oracle prédit la grille de départ ET la course. Analyst utilise la vraie grille (si disponible) et ne prédit que la course.",
        )
    use_real_grid = mode.startswith("🔬")

    if st.button("🏁 Lancer la prédiction", type="primary", width="stretch"):
        if not rounds_map or season not in rounds_map:
            st.error(f"Course « {gp} » introuvable pour la saison {season}.")
            return

        rnd = rounds_map[season]
        df = load_dataset()
        if df is None:
            st.error("Base de données introuvable (`f1_data_complete.csv`).")
            return

        with st.spinner(f"Entraînement des modèles & simulation — {official} {season}…"):
            with capture_logs() as buf:
                try:
                    results = train_and_predict(df, season, rnd, official, use_real_grid=use_real_grid)
                except Exception as e:
                    st.error(f"Erreur pendant la prédiction : {e}")
                    st.code(buf.getvalue())
                    return
            logs = buf.getvalue()

        if results is None or results.empty:
            st.error("Aucune prédiction générée.")
            st.code(logs)
            return

        grid_label = "Grille réelle" if use_real_grid else "Grille prédite (IA)"
        st.caption(f"Round {rnd} · {grid_label}")
        render_results(results)
        with st.expander("📟 Logs du moteur"):
            st.code(logs or "(vide)")


def render_results(results: pd.DataFrame) -> None:
    results = results.sort_values("Pos").reset_index(drop=True)

    # Podium (top 3)
    podium_html = '<div class="podium-wrap">'
    medals = {1: "p1", 2: "p2", 3: "p3"}
    labels = {1: "🥇 P1", 2: "🥈 P2", 3: "🥉 P3"}
    for _, r in results.head(3).iterrows():
        pos = int(r["Pos"])
        col = team_color(r["Team"])
        d = int(r["Delta"])
        delta_html = delta_span(d)
        podium_html += (
            f'<div class="podium {medals[pos]}">'
            f'<div class="pos">{labels[pos]}</div>'
            f'<div class="name">{r["DriverName"]}</div>'
            f'<div class="team" style="color:{col};">● {r["Team"]}</div>'
            f'<div class="delta">Grille P{int(r["Grid"])} · {delta_html}</div>'
            f'</div>'
        )
    podium_html += "</div>"
    st.markdown(podium_html, unsafe_allow_html=True)

    # Full table
    rows = ""
    for _, r in results.iterrows():
        pos = int(r["Pos"])
        badge_cls = f"p{pos}" if pos <= 3 else ""
        col = team_color(r["Team"])
        rows += (
            "<tr>"
            f'<td><span class="pos-badge {badge_cls}">{pos}</span></td>'
            f'<td><b>{r["DriverName"]}</b></td>'
            f'<td><span class="team-dot" style="background:{col};"></span>{r["Team"]}</td>'
            f'<td>P{int(r["Grid"])}</td>'
            f'<td>{delta_span(int(r["Delta"]))}</td>'
            "</tr>"
        )
    st.markdown(
        '<table class="delta-table"><thead><tr>'
        "<th>Pos</th><th>Pilote</th><th>Écurie</th><th>Grille</th><th>Δ</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


def delta_span(d: int) -> str:
    if d > 0:
        return f'<span class="up">▲ +{d}</span>'
    if d < 0:
        return f'<span class="down">▼ {d}</span>'
    return '<span class="flat">— 0</span>'


# ---------------------------------------------------------------------------
# Tab 2 — Data
# ---------------------------------------------------------------------------

def render_data_tab() -> None:
    st.subheader("État des données")
    status = [
        file_status_row("Résultats (Ergast)", RESULTS_CSV_PATH),
        file_status_row("Calendrier", CALENDAR_CSV_PATH),
        file_status_row("Télémétrie (FastF1)", EXTRA_CSV_PATH),
        file_status_row("Sprints", SPRINT_CSV_PATH),
        file_status_row("Dernière qualif", QUALI_CSV_PATH),
    ]
    st.dataframe(pd.DataFrame(status), width="stretch", hide_index=True)

    st.info("⚠️ Les mises à jour appellent les API Ergast/FastF1 et peuvent être **longues**. "
            "L'extraction FastF1 prend plusieurs **minutes** par saison.")

    current_year = datetime.now().year

    def run_update(label: str, fn, *args):
        with st.spinner(f"{label}…"):
            with capture_logs() as buf:
                try:
                    fn(*args)
                    ok = True
                except Exception as e:
                    ok = False
                    buf.write(f"\nERREUR: {e}\n")
            logs = buf.getvalue()
        load_calendar.clear()
        load_dataset.clear()
        (st.success if ok else st.error)(f"{label} — {'terminé' if ok else 'échec'}.")
        with st.expander("📟 Logs"):
            st.code(logs or "(vide)")

    tabs = st.tabs(["Résultats", "Calendrier", "FastF1", "Sprints", "Qualif"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        y1 = c1.number_input("Année début", 2001, current_year, 2001, key="res_y1")
        y2 = c2.number_input("Année fin", 2001, current_year, current_year, key="res_y2")
        if st.button("📥 Mettre à jour les résultats", width="stretch"):
            run_update("Mise à jour des résultats", update_database, int(y1), int(y2))

    with tabs[1]:
        c1, c2 = st.columns(2)
        y1 = c1.number_input("Année début", 2001, current_year, 2001, key="cal_y1")
        y2 = c2.number_input("Année fin", 2001, current_year, current_year, key="cal_y2")
        if st.button("📥 Mettre à jour le calendrier", width="stretch"):
            run_update("Mise à jour du calendrier", update_calendar, int(y1), int(y2))

    with tabs[2]:
        st.warning("⏱️ Très long : ~plusieurs minutes par saison (télémétrie FastF1, dispo dès 2018).")
        c1, c2 = st.columns(2)
        y1 = c1.number_input("Année début", 2018, current_year, current_year, key="ff_y1")
        y2 = c2.number_input("Année fin", 2018, current_year, current_year, key="ff_y2")
        if st.button("📥 Extraire les features FastF1", width="stretch"):
            run_update("Extraction FastF1", extract_fastf1_features, int(y1), int(y2))

    with tabs[3]:
        c1, c2 = st.columns(2)
        y1 = c1.number_input("Année début", 2021, current_year, 2021, key="sp_y1")
        y2 = c2.number_input("Année fin", 2021, current_year, current_year, key="sp_y2")
        if st.button("📥 Mettre à jour les sprints", width="stretch"):
            run_update("Mise à jour des sprints", update_sprint_data, int(y1), int(y2))

    with tabs[4]:
        c1, c2 = st.columns(2)
        y = c1.number_input("Année", 2001, current_year, current_year, key="q_y")
        rnd = c2.number_input("Round", 1, 30, 1, key="q_r")
        if st.button("📥 Récupérer la dernière qualif", width="stretch"):
            run_update("Mise à jour de la qualif", update_latest_qualifying, int(y), int(rnd))


# ---------------------------------------------------------------------------
# Tab 3 — Dev mode (season simulator)
# ---------------------------------------------------------------------------

def render_dev_tab() -> None:
    if not HAS_SIMULATOR:
        st.warning("Le simulateur de saison (`dev_tools/simulateur_saison.py`) est introuvable.")
        return

    st.caption("Backtest walk-forward d'une saison complète (outil de développement).")
    current_year = datetime.now().year
    c1, c2 = st.columns([1, 2])
    season = c1.number_input("Saison à simuler", 2001, current_year, 2024, step=1)
    mode = c2.radio("Mode", ["🔮 Oracle (grille prédite)", "🔬 Analyst (grille réelle)"], horizontal=True)
    use_real_grid = mode.startswith("🔬")

    if st.button("🧪 Lancer la simulation", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Initialisation…")

        def cb(done, total, msg):
            progress.progress(min(done / max(total, 1), 1.0), text=f"{msg} ({done}/{total})")

        with st.spinner(f"Simulation de la saison {int(season)}…"):
            with capture_logs() as buf:
                try:
                    summary = run_simulation(int(season), use_real_grid=use_real_grid, progress_callback=cb)
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.code(buf.getvalue())
                    return
            logs = buf.getvalue()
        progress.empty()

        if not summary:
            st.error("Aucune course évaluée (pas assez de données / historique).")
            st.code(logs)
            return

        st.subheader(f"Bilan {summary['season']} · {summary['mode']}")
        m = st.columns(5)
        m[0].metric("🏆 Vainqueur", f"{summary['winner_pct']:.1f}%")
        m[1].metric("Top 3", f"{summary['top3_pct']:.1f}%")
        m[2].metric("Top 5", f"{summary['top5_pct']:.1f}%")
        m[3].metric("Top 10", f"{summary['top10_pct']:.1f}%")
        m[4].metric("MAE", f"{summary['mae']:.2f}")
        st.caption(f"{summary['races_evaluated']} courses évaluées")

        rounds_df = pd.DataFrame(summary["rounds"])
        if not rounds_df.empty:
            rounds_df = rounds_df.rename(columns={
                "round": "Round", "winner_pred": "IA", "winner_real": "Réel",
                "winner_hit": "✓", "mae": "MAE", "top3": "Top3", "top5": "Top5", "top10": "Top10",
            })
            st.dataframe(rounds_df, width="stretch", hide_index=True)

        with st.expander("📟 Logs de la simulation"):
            st.code(logs or "(vide)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="The Delta Project", page_icon="🏎️", layout="wide")
    inject_css()
    header()

    tab_pred, tab_data, tab_dev = st.tabs(["🔮 Prédiction", "🔄 Données", "🛠️ Mode Dev"])
    with tab_pred:
        render_prediction_tab()
    with tab_data:
        render_data_tab()
    with tab_dev:
        render_dev_tab()


if __name__ == "__main__":
    main()
