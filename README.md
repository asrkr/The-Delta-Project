# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13.7-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.4_Stable-orange)

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
* **"Dual Brain" Architecture:** Two distinct models working in a chain:
    1.  **Qualifying Model:** Predicts the starting grid based on driver, team, and year.
    2.  **Race Model:** Predicts the final result using the grid (real or simulated).
* **Advanced Backtesting:** Full season simulator with scenario comparison:
    * *Oracle Mode:* The AI guesses everything (Qualif + Race).
    * *Analyst Mode:* The AI uses the real starting grid (isolates race performance).
* **Dynamic Management:** Automatic detection of participants (historical Entry List) and handling of transfers/rookies for future seasons (2025).

---

## 🛠️ Tech Stack

* **Language:** Python 3.13.7
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForestRegressor, LabelEncoder)
* **Data Collection:** Requests (REST API)

---

## 📂 Project Structure

```text
The-Delta-Project/
│
├── data/                      # CSV Storage (ignored by Git)
│   ├── f1_data_complete.csv   # Historical Database (2001-2025)
│   └── races_calendar.csv     # Official Calendar
│
├── src/                       # Source Code
│   ├── data_manager.py        # ETL Pipeline (Extract, Transform, Load)
│   ├── ml_model.py            # AI Brain (Training & Prediction Engine)
│   └── analysis.py            # Exploratory Data Analysis (Data Viz)
│
├── main.py                    # Script principal (Single Race Prediction)
├── simulateur_saison.py       # Backtesting Script (Full Season Simulation)
└── README.md                  # Documentation
````

-----

## ⚡ Installation & Usage

### 1\. Prerequisites

Clone the repo and install dependencies:

```bash
pip install pandas numpy scikit-learn requests seaborn matplotlib
```

### 2\. Data Initialization

Before first use, build the database:

```bash
python src/data_manager.py
# Choose option 2 to download full history (2001-2025)
```

### 3\. Run a Prediction (E.g., Abu Dhabi 2025)

```bash
python main.py
```

### 4\. Run a Season Simulation (Backtesting)

To test model accuracy on a past year (e.g., 2025):

```bash
python simulateur_saison.py
```

-----

## 🗺️ Roadmap

The project follows an iterative approach. Current status:

### ✅ Phase 1: Foundations (V1.4)

  - [x] Robust and incremental scraping (2001-2025).
  - [x] Pipeline ML modulaire (Random Forest).
  - [x] Gestion dynamique des participants et du Mercato.
  - [x] **V1.3:** Big Data History (from 2001) & Advanced Metrics.
  - [x] **V1.4:** Modular Architecture & Simulator Optimization.

### 🚧 Phase 2: Domain Intelligence (V1.5 - In Progress)

  - [ ] **Feature Engineering:** Calculation of "Recent Form" (rolling average).
  - [ ] **Circuit Analysis:** Track-specific qualifying impact.
  - [ ] **Dynamic Career:** Assessment of intrinsic driver level at time T.

### 🔮 Phase 3: Strategy & Environment (V2.0)

  - [ ] **Weather** integration.
  - [ ] **Pitstops** consideration (average time loss per track).
  - [ ] **Sprint** format handling.

### 🚀 Phase 4: Optimisation (V3.0)

  - [ ] **Model Swap:** Migration to **LightGBM Ranker** (Learning to Rank).
  - [ ] **Hyper-tuning:** Automated parameter optimization.

-----

## 📊 Current Performance (2025 Benchmark)

This section details the model's accuracy on the full 2025 season, comparing the two core prediction modes.

| Metric | 🔮 Oracle Mode (Predicted Grid) | 🔬 Analyst Mode (Real Grid) |
| :--- | :--- | :--- |
| **Winner Accuracy (P1)** | 34.8% | **69.6%** |
| **Top 3 Accuracy (Strict Order)** | 31.9% | 40.6% |
| **Top 5 Accuracy (Strict Order)** | 20.9% | 33.9% |
| **Top 10 Accuracy (Strict Order)** | 14.8% | 24.8% |
| **📉 Mean Absolute Error (MAE)** | 3.08 positions | **2.31 positions** |

-----

### Author

Project developed by an engineering student passionate about F1 and Computer Science.

-----
