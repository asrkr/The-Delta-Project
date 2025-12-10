# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/Model-RandomForest-purple?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.6_Stable-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**The Delta Project** is an Artificial Intelligence engine designed to predict Formula 1 race results.

The project leverages historical data (from 2001) and Machine Learning (**Random Forest**) to simulate a full race weekend: from qualifying to the chequered flag.

---

## 🏛️ Origin of the Name

Why **The Delta Project**?

The concept started with the idea of building an **Oracle** to predict race outcomes. Since “Oracle” is already very busy winning championships in F1, the project took inspiration from the most famous oracle of antiquity: the **Oracle of Delphi**.

The Greek initial for *Delphi* is **Delta** (Δ). It fits perfectly as a double meaning:

- A nod to the Oracle of Delphi  
- The mathematical symbol for **difference**, the core concept behind performance gaps and telemetry in motorsport

---

## 🚀 Key Features

- **Smart Data Pipeline**
  - Incremental downloading and merging of race results from **2001 to present** (via Jolpica / Ergast).
  - Non-destructive updates: only the requested seasons are refreshed.

- **Telemetry Integration (V1.6)**
  - Advanced race metrics via **FastF1**:
    - Average race pace
    - Best lap
    - Number of pitstops
    - Average time lost in pitstops
  - Stored in a dedicated `f1_extra_features.csv` file and merged into the main dataset.

- **“Dual Brain” Architecture**
  1. **Qualifying Model**  
     Predicts starting grid positions using:
     - Encoded driver, team, circuit
     - Rolling “form” on the grid (`form_grid`)
     - Career average grid positions (`career_grid_avg`)
     - Driver–circuit specific grid skill (`circuit_grid_skill`)
  2. **Race Model**  
     Predicts race finishing positions using:
     - Grid position (real or predicted)
     - Race form (`form_race`)
     - Career race averages (`career_race_avg`)
     - Pace telemetry (`career_race_pace`, `career_best_lap`, `career_pit_loss`)
     - Circuit-specific race skill (`circuit_race_skill`)

- **Advanced Backtesting**
  - Full-season simulator:
    - **Oracle Mode**: the AI predicts the grid and the race.
    - **Analyst Mode**: the AI receives the *real* starting grid and only predicts the race outcome.
  - Evaluation metrics:
    - Winner accuracy (P1)
    - Top 3 / Top 5 / Top 10 (strict order)
    - Mean Absolute Error (MAE) on predicted positions.

- **Dynamic Driver Management**
  - Automatic detection of race participants based on the historical entry list.
  - Handles transfers and rookies when simulating future seasons.

---

## 🧠 Model Philosophy

The Delta Project focuses on **performance modelling**, not randomness.

Assumptions:

- Race outcomes are driven by:
  - Driver skill
  - Car performance
  - Circuit characteristics
  - Strategy & execution (reflected through pace and pit metrics)
- Grid position is crucial, but its **impact depends on the circuit**:
  - Some tracks are “overtaking hell” (high grid → high finish correlation).
  - Others are more chaotic or strategic.

Deliberately **not** modelled:

- Safety cars
- Mechanical failures
- Crashes
- Weather randomness (for now)

The AI’s predictions should be interpreted as:

> “Most likely finishing order **if nothing crazy happens** and everyone runs to form.”

---

## 🛠️ Tech Stack

- **Language:** Python 3.13+
- **Data:** Pandas, NumPy
- **Machine Learning:** scikit-learn (`RandomForestRegressor`, `LabelEncoder`)
- **Data Collection:**
  - `requests` (REST API Jolpica/Ergast)
  - `fastf1` (timing & telemetry)

---

## 📂 Project Structure

```text
The-Delta-Project/
│
├── src/                          # Core source code
│   ├── data_manager.py           # ETL Pipeline (Ergast + FastF1 + calendar)
│   └── ml_model.py               # Feature engineering & ML models (qualif + race)
│
├── main.py                       # Main entry point (single race prediction)
├── update_manager.py             # Maintenance script (update/refresh datasets)
│
└── README.md                     # You are here
````

---

## ⚡ Installation & Usage

### 1. Clone & Install Dependencies

```bash
pip install pandas numpy scikit-learn requests fastf1 tqdm matplotlib seaborn
```

*(Some libraries like `matplotlib` / `seaborn` are mainly used in dev tools & plotting.)*

---

### 2. Initialise / Update the Data

All data updates are handled via `update_data.py` (a small menu-based manager).

```bash
python update_manager.py
```

Typical first-time setup:

1. **Update Ergast results** (2001 → current year)
2. **Update calendar** (to get `races_calendar.csv`)
3. **Extract FastF1 features** (telemetry for the seasons you care about)

Depending on what you’ve implemented in your menu, it’s usually something like “Option 7: Update ALL”.

This generates:

* `data/f1_data_complete.csv`
* `data/races_calendar.csv`
* `data/f1_extra_features.csv`

---

### 3. Run a Single-Race Prediction

Example: predict Abu Dhabi 2025.

```bash
python main.py
```

The script will:

1. Load `f1_data_complete.csv`
2. Ask which Grand Prix you want (partial name works, e.g. `"Abu Dhabi"`)
3. Ask which season (e.g. `2025`)
4. Ask whether to use the real grid if available (`o/n`)
5. Train the models on all races **before** that GP
6. Print the predicted results table:

* Predicted grid (if `use_real_grid=False`)
* Predicted race order
* Δ between grid and final position

---

### 4. Run a Season Simulation (Dev / Benchmark)

The backtesting simulator is in `dev_tools/simulateur_saison.py`.

To evaluate a full season (for example 2025):

```bash
python dev_tools/simulateur_saison.py
```

You’ll be asked:

* Which season to simulate
* Whether to use real grids or predicted grids

At the end, you get metrics:

* Winner accuracy (%)
* Top 3 / Top 5 / Top 10 strict-order accuracy
* Average MAE on positions

---

## 🗺️ Roadmap

The project follows an iterative approach.
**Current status: V1.6 – Telemetry.**

### ✅ Phase 1: Foundations (V1.4)

* [x] Robust and incremental scraping (2001–2025)
* [x] Functional ML pipeline (Random Forest)
* [x] Dynamic participant retrieval per race
* [x] Season simulator with MAE & Top-K metrics

### ✅ Phase 2: Domain Intelligence (V1.5)

* [x] **Feature Engineering:** recent form (3-race rolling averages)
* [x] **Circuit analysis:** correlation between grid & result per circuit (`circuit_importance`)
* [x] **Career stats:** overall averages and circuit-specific skills for each driver
* [x] **Initial hyperparameter tuning:** Random Forest optimisation

### ✅ Phase 3: Strategy & Environment (V1.6 – Telemetry)

* [x] **Telemetry Integration (FastF1)**:

  * Average pace over the race
  * Best lap
  * Pitstop count & mean pit loss
* [x] **Career telemetry stats:** `career_race_pace`, `career_best_lap`, `career_pit_loss`
* [x] **Real grid injection:** ability to load and use the actual qualifying result (`latest_qualifying.csv`)
* [x] **Driver identity hardening:** `DriverKey` harmonised between Ergast and FastF1 (e.g. `m_verstappen`, `j_verstappen`)
* [x] **Season-level pace ranking:** `pace_rank_season` (relative pace ranking within a season)
* [x] **Global hyperparameter tuner** for the full pipeline (`dev_tools/hyperparameter_pipeline_tuning.py`)
* [ ] Weather integration
* [ ] Sprint format modelling

### 🚀 Phase 4: Next-Gen Models (V2.x)

Planned for the next major version (beyond V1.6):

* [ ] Swap Random Forest → **Gradient Boosting** models:

  * LightGBM / XGBoost
  * Or CatBoost for heterogeneous tabular data
* [ ] Dedicated learning-to-rank model for qualifying
* [ ] Probabilistic outputs (distributions over finishing positions instead of single-point estimates)
* [ ] More explicit separation between:

  * **Pace model**
  * **Race-position model**
  * **Risk / volatility model**

---

## 📊 Current Performance (Reference Season – 2025, V1.6)

| Metric      | 🔮 Oracle Mode | 🔬 Analyst Mode |
| ----------- | -------------- | --------------- |
| Winner (P1) | 29,2%          | 65,2%           |
| Top 3       | 20,8%          | 43,1%           |
| Top 5       | 15,8%          | 35%             |
| Top 10      | 13,8%          | 23,8%           |
| MAE         | 4.16           | 3,42            |

### Interpretation

* Once the **starting grid is known**, the race model is very strong:
  it converts the grid + telemetry profile into a realistic finishing order.
* The main bottleneck of the global system is now the **qualifying model**:
  improving grid prediction will directly translate into better Oracle Mode performance.
* Telemetry features have improved the **race understanding** (especially in Analyst Mode),
  even though some earlier experiments (over-normalising grid or over-weighting contextual deltas)
  temporarily degraded performance — those changes have been rolled back in the final V1.6.

---

## 👨‍💻 Author

The Delta Project is developed by an engineering student passionate about Formula 1 and Computer Science.

---

## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute the code, provided that the original copyright notice is retained.

See the [LICENSE](LICENSE) file for more details.
