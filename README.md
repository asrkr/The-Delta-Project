# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13.7-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.6_Telemetry-orange)

**The Delta Project** is an Artificial Intelligence engine designed to predict Formula 1 race results.

This project leverages historical data (scraped from 2001) and Machine Learning algorithms (**Random Forest**) to simulate a full race weekend: from the qualifying session down to the chequered flag.

---

## 🏛️ Origin of the Name

Why **The Delta Project**?

The concept started with the idea of building an **Oracle** to predict race outcomes. However, since "Oracle" is already quite busy winning championships in F1, we turned to the most famous oracle of antiquity: the **Oracle of Delphi**.

The Greek initial for Delphi is **Delta** ($\Delta$). It fits perfectly as a double entendre: a nod to the Oracle and the mathematical symbol for *difference*, the core of F1 telemetry.

---

## 🚀 Key Features

* **Smart Data Pipeline:** Incremental downloading and cleaning of data from the 2001 season to the present day (via Jolpica/Ergast API).
* **Telemetry Integration (V1.6):** Injection of advanced race metrics via **FastF1** (Average Race Pace, Best Lap, Pitstop Loss) to understand car performance beyond simple results.
* **"Dual Brain" Architecture:** Two distinct models working in a chain:
    1.  **Qualifying Model:** Predicts the starting grid based on driver, team, and year.
    2.  **Race Model:** Predicts the final result using the grid (real or simulated) and telemetry profile.
* **Advanced Backtesting:** Full season simulator with scenario comparison:
    * *Oracle Mode:* The AI guesses everything (Qualif + Race).
    * *Analyst Mode:* The AI uses the real starting grid (isolates race performance).
* **Dynamic Management:** Automatic detection of participants (historical Entry List) and handling of transfers/rookies for future seasons.

---

## 🧠 Model Philosophy

The Delta Project is designed to **model F1 performance, not randomness**.

The system assumes that:
- Race outcomes are primarily driven by **driver skill, car performance, track characteristics, and strategic efficiency**.
- Grid position is a critical but **not absolute determinant**, depending on circuit-specific constraints.
- Telemetry data (pace, degradation, pit loss) provides a **truer signal of race execution** than finishing positions alone.

The model deliberately avoids:
- Lap-by-lap simulation.
- Stochastic events (crashes, safety cars, mechanical DNFs).
- Short-term “noise-driven” correlations.

Instead, The Delta Project focuses on **learning stable performance deltas** (`Δ`) between drivers and teams across races, seasons, and circuits.

Predictions should therefore be interpreted as:
> *“Most likely outcome given equal conditions and no major external disruption.”*

---

## 🛠️ Tech Stack

* **Language:** Python 3.13.7
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForestRegressor, LabelEncoder)
* **Data Collection:** Requests (REST API), FastF1

---

## 📂 Project Structure

```text
The-Delta-Project/
│
├── src/                       # Source Code
│   ├── data_manager.py        # ETL Pipeline (Extract, Transform, Load)
│   └── ml_model.py            # AI Brain (Feature Engineering & Prediction)
│
├── main.py                    # Main Script (Single Race Prediction)
├── update_data.py             # Maintenance Script (Update Database)
└── README.md                  # Documentation
````

-----

## ⚡ Installation & Usage

### 1\. Prerequisites

Clone the repo and install dependencies:

```bash
pip install pandas numpy scikit-learn requests seaborn matplotlib fastf1
```

### 2\. Data Initialization

Before first use, build the database using the update manager:

```bash
python update_manager.py
# Select Option 7 (Update ALL) for a fresh start
```

### 3\. Run a Prediction (E.g., Abu Dhabi 2025)

```bash
python main.py
```

### 4\. Run a Season Simulation (Dev Only)

To test model accuracy on a past year (e.g., 2025):

```bash
python dev_tools/simulateur_saison.py
```

-----

## 🗺️ Roadmap

The project follows an iterative approach. Current status: **V1.6 Telemetry**.

### ✅ Phase 1: Foundations (V1.4)

  - [x] Robust and incremental scraping (2001-2025).
  - [x] Functional ML Pipeline (Random Forest).
  - [x] Dynamic Participant Retrieval.
  - [x] Precision Metrics (MAE, Top 3, Top 10).

### ✅ Phase 2: Domain Intelligence (V1.5)

  - [x] **Feature Engineering:** Calculation of "Recent Form" (rolling average).
  - [x] **Circuit Analysis:** Calculation of track-specific grid impact (Correlation).
  - [x] **Full Career Analysis:** Integration of career stats and track-specific skills.
  - [x] **Hyper-tuning:** Automated parameter optimization.

### 🚧 Phase 3: Strategy & Environment (V1.6 - In Progress)

  - [x] **Telemetry Integration:** Average Race Pace & Best Lap analysis (FastF1).
  - [x] **Pitstops Analysis:** Calculation of average time lost in pits per race.
  - [x] **Real Grid Injection:** Ability to use real qualifying results for race prediction.
  - [ ] **Weather** integration.
  - [ ] **Sprint** format handling.

### 🚀 Phase 4: Optimization (V3.0)

  - [ ] **Model Swap:** Migration to **LightGBM Ranker** (Learning to Rank).

-----

## 📊 Current Performance (2025 Benchmark)

This section details the model's accuracy on the full 2025 season (**V1.6 Telemetry**), comparing the two core prediction modes.

| Metric | 🔮 Oracle Mode (Predicted Grid) | 🔬 Analyst Mode (Real Grid) |
| :--- | :--- | :--- |
| **Winner Accuracy (P1)** | 34.8% | **69.6%** |
| **Top 3 Accuracy (Strict Order)** | 31.9% | 40.6% |
| **Top 5 Accuracy (Strict Order)** | 20.9% | 33.9% |
| **Top 10 Accuracy (Strict Order)** | 14.8% | 24.8% |
| **📉 Mean Absolute Error (MAE)** | 3.08 positions | **2.31 positions** |

### Interpretation

The **V1.6** update introduces complex telemetry data (Race Pace, Pit Loss). While the Oracle Mode accuracy has stabilized around 35%, the **Analyst Mode** (knowing the grid) reaches a massive **69.6%** winner accuracy, proving that the Race Model perfectly understands car performance when the starting position is known. The focus now returns to improving the Qualifying Model to bridge the gap between the two modes.

-----

### Author

Project developed by an engineering student passionate about F1 and Computer Science.
