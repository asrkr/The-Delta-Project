# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.13.7-blue?style=flat\&logo=python)
![Machine Learning](https://img.shields.io/badge/AI-RandomForest-green?style=flat\&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-V1.6_Telemetry-orange)

**The Delta Project** is an Artificial Intelligence engine designed to predict Formula 1 race results.

The system leverages historical race data (2001–present) and machine learning to model **performance deltas** between drivers, teams, and circuits, simulating a full race weekend: qualifying → race.

---

## 🏛️ Origin of the Name

The project was initially conceived as an **oracle** predicting race outcomes. Since F1 already has its share of dominant oracles, the name was inspired by the **Oracle of Delphi**.

The Greek initial for Delphi is **Delta (Δ)** — a symbol that also represents *difference*, central to both telemetry analysis and performance modeling.

---

## 🚀 Key Features

### Core Capabilities

* **Incremental Data Pipeline**

  * Historical data from 2001 onward (Ergast / Jolpica APIs)
  * Automatic handling of rookies, transfers, and team changes

* **Dual‑Model Architecture**

  1. **Qualifying Model** → Predicts starting grid
  2. **Race Model** → Predicts final positions based on grid + performance context

* **Telemetry Integration (V1.6)**

  * Average race pace
  * Best lap
  * Pit‑stop loss estimation
  * Long‑run consistency indicators

* **Dynamic Simulation Modes**

  * **Oracle Mode** → Grid + Race fully predicted
  * **Analyst Mode** → Real grid injected, race performance isolated

* **Season‑Scale Backtesting**

  * Walk‑forward simulation
  * Strict Top‑N accuracy metrics
  * Mean Absolute Error (MAE)

---

## 🧠 Modeling Philosophy

The Delta Project focuses on **structural performance**, not race randomness.

Assumptions:

* Results emerge primarily from

  * driver skill
  * car performance
  * circuit characteristics
  * execution quality
* Grid position is influential but **contextual**, not absolute

Deliberately excluded:

* Safety cars & random incidents
* Lap‑by‑lap stochastic simulation
* One‑off chaotic race events

Predictions should be interpreted as:

> *Most likely finishing order under equal and stable conditions.*

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Data:** Pandas, NumPy
* **ML:** Scikit‑learn (RandomForestRegressor)
* **Telemetry:** FastF1

---

## 📂 Project Structure

```text
The-Delta-Project/
[
├── src/
│   ├── data_manager.py        # Data ingestion & cleaning
│   ├── ml_model.py            # Feature engineering + models
|
├── main.py                    # Single race prediction
├── update_data.py             # Dataset refresh utility
└── README.md
```

---

## ⚡ Installation & Usage

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn requests fastf1
```

### 2. Build / update dataset

```bash
python update_manager.py
# Recommended: full refresh for first run
```

### 3. Predict a single race

```bash
python main.py
```

### 4. Backtest a full season

```bash
python dev_tools/simulateur_saison.py
```

---

## 📊 Performance Benchmark — V1.6 (2025 Season)

| Metric      | 🔮 Oracle Mode | 🔬 Analyst Mode |
| ----------- | -------------- | --------------- |
| Winner (P1) | 29,2%          | 65,2%           |
| Top 3       | 20,8%          | 43,1%           |
| Top 5       | 15,8%          | 35%             |
| Top 10      | 13,8%          | 23,8%           |
| MAE         | 4.16           | 3,42            |

**Interpretation**

* Race model strongly captures **true car performance** when the grid is known
* Remaining limitation lies in qualifying prediction quality
* Telemetry features improve stability, not peak accuracy

---

## 🧊 Version Freeze — V1.6 Decision

V1.6 is now **frozen** with the following design choices:

✅ Grid retained as a race feature (contextual importance)
✅ Telemetry features preserved but conservatively weighted
✅ DriverKey normalization stabilized
✅ RandomForest retained for interpretability & robustness

No further feature additions will be made to V1.6.

---

## 🔮 Roadmap (Next Major Version)

**V2.0 — Performance Ranking Era**

Planned changes:

* Learning‑to‑Rank model for qualifying
* Gradient Boosting (LightGBM / CatBoost)
* Probability distributions instead of point estimates
* Explicit uncertainty modeling

V2.0 will intentionally **break compatibility** with V1.x.

---

## ✍️ Author

Personal research project developed by an engineering student passionate about Formula 1, telemetry, and machine learning.
