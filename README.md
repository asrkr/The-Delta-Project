# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python)
![Machine
Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.3_Stable-orange)

## Overview

**The Delta Project** is a professional-grade Artificial Intelligence
engine designed to predict Formula 1 race results using historical data,
advanced preprocessing, and machine learning models.

This system simulates an entire race weekend---from qualifying to the
chequered flag---leveraging a Random Forest pipeline trained on over two
decades of Formula 1 history.

------------------------------------------------------------------------

## ✨ Features

-   **Automated Data Pipeline**\
    Incremental extraction, cleaning, and caching of race data
    (2001--present).

-   **Two-Stage Machine Learning Architecture**

    -   *Qualifying Model* --- Predicts starting grid.\
    -   *Race Model* --- Predicts final classification using simulated
        or real grid.

-   **Season Simulator (Backtesting)**

    -   *Oracle Mode*: AI predicts quali + race.\
    -   *Analyst Mode*: AI uses real grids to isolate race performance.

-   **Intelligent Driver & Team Management**\
    Automatic handling of rookies, transfers, and multi-team seasons.

------------------------------------------------------------------------

## 📂 Project Structure

``` text
The-Delta-Project/
│
├── data/                      # Local dataset storage
│   ├── f1_data_complete.csv
│   └── races_calendar.csv
│
├── src/
│   ├── data_manager.py        # ETL pipeline
│   ├── ml_model.py            # ML logic
│   └── analysis.py            # Analytics & visualisations
│
├── main.py                    # Single-race prediction script
├── simulateur_saison.py       # Backtesting engine
└── README.md
```

------------------------------------------------------------------------

## ⚡ Installation

### Install dependencies

``` bash
pip install pandas numpy scikit-learn requests seaborn matplotlib
```

### Initialize dataset

``` bash
python src/data_manager.py
# Choose option 2 (full 2001–2025 history)
```

### Run a prediction

``` bash
python main.py
```

### Run a season simulation

``` bash
python simulateur_saison.py
```

------------------------------------------------------------------------

## 🗺 Roadmap

### ✔ Phase 1 --- Foundations (V1.3)

-   Robust data ingestion
-   ML pipeline (Random Forest)
-   Driver transfer logic
-   Quality metrics (MAE, accuracy)

### 🚧 Phase 2 --- Intelligence Layer (V1.5)

-   Driver recent form (rolling windows)
-   Track-dependent quali bias
-   Career progression model

### 🔮 Phase 3 --- Strategy Layer (V2.0)

-   Weather modelling
-   Pitstop strategy integration
-   Sprint race support

------------------------------------------------------------------------

## 📊 Current Performance (2024)

-   **Winner Accuracy:** \~35--40%\
-   **Top 10 Accuracy:** \~65%\
-   **MAE:** \~3.5 positions

------------------------------------------------------------------------

## 👨‍💻 Author

Developed by an engineering student passionate about **AI**, **Data
Science**, and **Formula 1**.
