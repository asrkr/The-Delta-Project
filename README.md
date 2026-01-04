# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/Model-RandomForest-purple?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.8_Context_%26_Weather-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**The Delta Project** is an Artificial Intelligence engine designed to predict Formula 1 race results.

The project leverages historical data (from 2001) and Machine Learning (**Random Forest**) to simulate a full race weekend: from qualifying to the chequered flag.

> **V1.8** extends the V1.7.1 stability work by introducing **clean-air pace** and **weather/temperature race context**
> while keeping **strict temporal integrity** across training, internal benchmarking tools, and hyperparameter tuning.

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

- **Telemetry Integration (V1.6 → V1.8)**
  - Advanced race metrics via **FastF1** (telemetry available from **2018+**):
    - Average race pace
    - **Clean-air pace** (V1.8)
    - Best lap
    - Number of pitstops
    - Average time lost in pitstops
    - **Race-day weather context**: `is_rainy`, `track_temp` (V1.8)
  - Stored in a dedicated `f1_extra_features.csv` file and merged into the main dataset.

- **Sprint Weekend Context (V1.7)**
  - Sprint results extracted via **Ergast API** and stored in a dedicated dataset.
  - Sprint data is treated as **context**, not as a prediction target.
  - Sprint signals enrich race understanding without altering the core model logic:
    - Explicit sprint-weekend flag
    - Relative performance delta between Sprint grid and finish
  - No impact on non-sprint weekends (hard separation).

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
     - Pace telemetry (`career_race_pace`, `career_clean_air_pace`, `career_best_lap`, `career_pit_loss`) (V1.8)
     - Circuit-specific race skill (`circuit_race_skill`)
     - Sprint contextual signals (when applicable)
     - **Weather context + wet skill** (`is_rainy`, `track_temp`, `career_wet_skill`) (V1.8)

- **Pipeline Integrity & Validation (V1.7.1 → V1.8)**
  - Strict time-based evaluation: **no future race data** is used for feature computation, training, or tuning.
  - Canonical driver identity handling (`DriverKey`) to prevent merge mismatches across sources (Ergast / FastF1 / sprint datasets).
  - Walk-forward hyperparameter tuning aligned with real-world usage (train on past races → validate on the next race).
  - Forward-style validation on tricky seasons (regulation changes / mixed formats / limited history).

- **Advanced Backtesting (Internal Tool)**
  - Full-season simulator:
    - **Oracle Mode**: the AI predicts the grid and the race.
    - **Analyst Mode**: the AI receives the *real* starting grid and only predicts the race outcome.
  - Evaluation metrics:
    - Winner accuracy (P1)
    - Top 3 / Top 5 / Top 10 (strict order)
    - Mean Absolute Error (MAE) on predicted positions.
  - Sprint-aware benchmarks (Sprint vs Non-Sprint weekends).
  > Note: this simulator is a **personal/internal development tool** and is **not included** in the public repository.

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
  - Some tracks are “overtaking hell”.
  - Others allow significant position swings.

Sprint races are considered as:
> “High-signal short-format race context, useful to refine Sunday expectations.”

Deliberately **not** modelled:
- Safety cars
- Mechanical failures
- Crashes
- Fully stochastic race chaos

Weather is handled as **explicit race context** (rain flag + track temperature), not as a full stochastic simulator.

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
│   ├── data_manager.py           # ETL Pipeline (Ergast + FastF1 + calendar + sprints)
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
pip install pandas numpy scikit-learn requests fastf1
```

---

### 2. Initialise / Update the Data

All data updates are handled via `update_manager.py`.

```bash
python update_manager.py
```

Typical first-time setup:

1. Update Ergast results
2. Update calendar
3. Extract FastF1 telemetry
4. Extract Sprint results (V1.7)

This generates:

* `data/f1_data_complete.csv`
* `data/races_calendar.csv`
* `data/f1_extra_features.csv`
* `data/f1_sprint_results.csv`

---

### 3. Run a Single-Race Prediction

```bash
python main.py
```

---

### 4. Run a Season Simulation (Dev / Benchmark)

Season simulation is performed using an **internal development tool** (not included in the public repository).
Sprint-aware benchmarks can be run using dedicated internal benchmark scripts.

---

## 🗺️ Roadmap

**Current status: V1.8 – Context & Weather Release.**

### ✅ Phase 4: Context & Robustness (V1.8)

* [x] Clean-air pace integration (`clean_air_pace` → `career_clean_air_pace`)
* [x] Race-day weather context (`is_rainy`, `track_temp`)
* [x] Wet performance context (`career_wet_skill`)
* [x] Multi-season stress testing (2020 / 2021 / 2022)
* [x] Forward-style validation (2025)

### ✅ Phase 3.5: Pipeline Stabilization (V1.7.1)

* [x] Temporal leak removal (training / internal simulation / tuning)
* [x] Walk-forward evaluation alignment
* [x] Canonical driver identity handling (`DriverKey`) across sources

### ✅ Phase 3: Strategy & Environment (V1.6–V1.7)

* [x] Telemetry Integration (FastF1)
* [x] Career telemetry aggregation
* [x] Real grid injection
* [x] Robust driver identity handling (`DriverKey`)
* [x] Season-level pace normalization
* [x] Sprint weekend contextual integration
* [x] Sprint-aware benchmarking and validation

### 🚀 Phase 5: Next-Gen Models (V2.x)

* Learning-to-Rank for qualifying
* Gradient Boosting (LightGBM / CatBoost)
* Probabilistic race outcome distributions
* Explicit separation between pace, position, and variance modelling

---

## 📊 Current Performance (Reference Season – 2025, V1.8)

| Metric      | 🔮 Oracle Mode | 🔬 Analyst Mode |
| ----------- | -------------- | --------------- |
| Winner (P1) | 20,8%          | 66,7%           |
| Top 3       | 16,7%          | 45,8%           |
| Top 5       | 13,3%          | 40,0%           |
| Top 10      | 12,9%          | 26,2%           |
| MAE         | 3,08           | 2,39            |

> Analyst Mode isolates race modelling using the real starting grid.
> Oracle Mode evaluates the full predictive pipeline (qualifying + race).
>
> Forward Oracle results are intentionally lower and reflect realistic uncertainty.

---

## 👨‍💻 Author

The Delta Project is developed by an engineering student passionate about Formula 1 and Computer Science.

---

## 📄 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute the code, provided that the original copyright notice is retained.

See the [LICENSE](LICENSE) file for more details.
