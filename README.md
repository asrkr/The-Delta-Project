# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13.7-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.3_Stable-orange)

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
    * *Oracle Mode:* The AI guesses everything (Quali + Race).
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
│   ├── ml_model.py            # AI Brain (Training & Prediction)
│   └── analysis.py            # Exploratory Data Analysis (Data Viz)
│
├── main.py                    # Main Script (Single Race Prediction)
├── simulateur_saison.py       # Backtesting Script (Full Season Simulation)
└── README.md                  # Documentation
