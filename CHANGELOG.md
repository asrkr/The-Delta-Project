# Changelog — The Delta Project

All notable changes to this project are documented in this file.  
The project follows an **iterative, benchmark-driven development approach**.

---

## [v1.6] — Telemetry & Robustness (Frozen)

### Added
- Integration of **FastF1 telemetry data**:
  - Average race pace
  - Best lap time
  - Pit stop count
  - Mean pit time loss
- Advanced feature engineering pipeline
- Full **driver identity unification** between Ergast & FastF1:
  - Normalized `DriverKey`
  - Robust handling of historical name collisions (e.g. Verstappen, Schumacher)
- Optional **real qualifying grid injection** for race prediction
- Full **season walk-forward simulation pipeline**
- Automatic feature importance analysis
- End-to-end hyperparameter tuning via pipeline tuner

### Machine Learning
- Core model: `RandomForestRegressor`
- Decoupled models:
  - Qualifying prediction
  - Race outcome prediction
- Experimental features tested:
  - `grid_delta`
  - grid normalization
  - `pace_rank_season`
  - `expected_race_rank` / contextual grid deltas
- Systematic evaluation of each feature via season benchmarks

### Results (reference season)
- **IA-only (no real grid)**
  - Winner accuracy ≈ 29–33%
  - MAE ≈ 4.1
- **With real grid**
  - Winner accuracy ≈ 58–63%
  - Top 3 ≈ 43–46%
  - MAE ≈ 3.4

**Conclusion:**  
Race grid position remains the dominant variable in Formula 1.  
The v1.6 model is now **robust, explainable and stable**, but bounded by the limitations of RandomForest on ranking-heavy problems.

### Known limitations
- Strong dependency on grid whenever available
- IA-only performance slightly below v1.5
- RandomForest limitations:
  - poor learning-to-rank behavior
  - no probabilistic output
  - limited interaction modeling

👉 Version **v1.6 is frozen** and serves as the stable baseline for the next major iteration.

---

## [v1.5] — Domain Intelligence

### Added
- Driver recent form (rolling average)
- Career-long driver statistics
- Circuit-specific driver skills
- Circuit grid impact estimation
- First automated hyperparameter tuning

### Notes
- Strong performance with real grids
- Reduced robustness in unseen conditions

---

## [v1.4] — Foundations

### Added
- Full Ergast scraping (2001–2025)
- End-to-end ML pipeline
- Race-by-race simulation
- Advanced evaluation metrics:
  - MAE
  - Top 3 / Top 5 / Top 10
- Modular project architecture

---
