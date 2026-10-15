# 🏎️ The Delta Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)

**The Delta Project** is an rtificial intelligence engine designed to predict Formula 1 race results.

The project leverages historical data (from 2001) and machine learning to simulate a full race week: from qualifying to the chequered flag.


## 🏛️ Origin of the name

Why **The Delta Project**?

The concept started with the idea of building an **Oracle** to predict race outcomes. Since "Oracle" is already very busy (among other things) being a title sponsor for one of the most competitive F1 team in recent years, the project took inspiration from the most famous oracle of antiquity: the **Oracle of Delphi**.

The Greek letter for *Delphi* is **Delta** (Δ). It fits perfectly as a double meaning:
- A nod to the Oracle of Delphi
- The mathematical symbol for **difference**, the core concept behind performance gaps and telemetry in motorsport

## 📦 Releases

### [v0.1.0] - Data Engine & ETL Pipeline
- Canonical driver and team key normalisation (`keys.py`)
- Ergast / Jolpica API REST connectors (`ergast.py`, `sprint.py`, `calendar.py`)
- Incremental database storage & orchestration (`manager.py`)
- Unit tests battery using `pytest`