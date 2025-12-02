# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13.7-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.5_Stable-orange)

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
* **Feature Engineering (V1.5):** Advanced logic to feed the AI with context:
    * *Recent Form:* 3-race rolling average (Qualifying & Race pace).
    * *Circuit Impact:* Historical correlation of the track (Procession vs Overtaking friendly).
    * *Career Profile:* Intrinsic driver performance level at the time of the race.
* **Advanced Backtesting:** Full season simulator with scenario comparison (Oracle vs Analyst modes) and strict accuracy metrics.
* **Dynamic Management:** Automatic detection of participants and handling of future seasons via fallback logic.

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
│   ├── ml_model.py            # AI Brain (Feature Engineering & Prediction)
│   └── analysis.py            # Exploratory Data Analysis (Data Viz)
│
├── main.py                    # Main Script (Single Race Prediction)
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

The project follows an iterative approach. Current status: **V1.5 Stable**.

### ✅ Phase 1: Foundations (V1.4)

  - [x] Robust and incremental scraping (2001-2025).
  - [x] Functional ML Pipeline (Random Forest).
  - [x] Dynamic Participant Retrieval.
  - [x] Precision Metrics (MAE, Top 3, Top 10).

### ✅ Phase 2: Domain Intelligence (V1.5)

  - [x] **Recent Form:** Implementation of rolling averages for Grid and Race positions.
  - [x] **Circuit Analysis:** Calculation of track-specific grid impact (Correlation).
  - [x] **Full Career Analysis:** Integration of career stats and track-specific skills.

### 🔮 Phase 3: Strategy & Environment (V2.0)

  - [ ] **Weather** integration.
  - [ ] **Pitstops** consideration (average time loss per track).
  - [ ] **Sprint** format handling.

### 🚀 Phase 4: Optimization (V3.0)

  - [ ] **Model Swap:** Migration to **LightGBM Ranker**.
  - [ ] **Hyper-tuning:** Automated parameter optimization.

-----

## 📊 Current Performance (2025 Benchmark)

This section details the model's accuracy on the full 2025 season (V1.5 Full Features), comparing the two core prediction modes.

| Metric | 🔮 Oracle Mode (Predicted Grid) | 🔬 Analyst Mode (Real Grid) |
| :--- | :--- | :--- |
| **Winner Accuracy (P1)** | **39.1%** | **56.5%** |
| **Top 3 Accuracy (Strict Order)** | 27.5% | 39.1% |
| **Top 5 Accuracy (Strict Order)** | 21.7% | 35.7% |
| **Top 10 Accuracy (Strict Order)** | 15.7% | 27.0% |
| **📉 Mean Absolute Error (MAE)** | 3.09 positions | **2.30 positions** |

### Interpretation

The **V1.5** update successfully improved the "Oracle" capabilities, reaching nearly **40% winner accuracy** without knowing the grid. The Drop in Analyst Mode metrics compared to V1.4 suggests the model is becoming more complex and nuanced, relying less on raw grid position and more on driver/car dynamics.

-----

### Author

Project developed by an engineering student passionate about F1 and Computer Science.
