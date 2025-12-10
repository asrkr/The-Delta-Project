# 📜 Changelog

All notable changes to **The Delta Project** are documented in this file.

The project follows a versioned, iterative development approach focused on
model stability, domain intelligence, and progressive realism.

---

## [v1.6.0] – Telemetry Integration  
**Release date:** 2025

### ✨ Added
- **FastF1 telemetry integration**:
  - Average race pace
  - Best lap time
  - Pit stop loss estimation
- **Race execution awareness** through telemetry-derived features.
- **Real grid injection**:
  - Ability to run race predictions using real qualifying results.
  - Enables clear separation between qualifying accuracy and race modeling quality.
- **Season-level pace normalization** (`pace_rank_season`).
- **Improved driver identity handling**:
  - Stable `DriverKey` generation to avoid name collisions (e.g. Verstappen, Schumacher).

---

### 🔄 Changed
- **Race Model feature set rebalanced**:
  - Grid-related contextual features were evaluated and simplified for stability.
  - Telemetry features prioritized over over-engineered grid transformations.
- **Qualifying ↔ Race decoupling reinforced**:
  - Qualifying predicts grid only.
  - Race model focuses on execution and pace conditional on starting position.
- **Hyperparameter tuning pipeline updated** to reflect the new feature space.
- **Model benchmarks redefined** using two explicit scenarios:
  - *Oracle Mode* (predicted grid)
  - *Analyst Mode* (real grid)

---

### 🗑️ Removed
- Experimental grid normalization variants that degraded generalization:
  - `grid_z`
  - `grid_percent`
- Overly synthetic grid deltas that did not improve Oracle Mode performance.

---

### 🐞 Fixed
- Driver name / team mismatches causing label encoder crashes.
- Rookie / mid-season team edge cases.
- Silent data leakage between training and prediction phases.

---

### ⚠️ Known Limitations
- Oracle Mode (full AI prediction) remains limited by qualifying model accuracy.
- Stochastic race events are intentionally not modeled:
  - DNFs
  - Safety Cars
  - Weather randomness
- Sprint formats not yet supported.

---

## Previous Versions

### [v1.5] – Domain Intelligence
- Recent form (rolling averages)
- Circuit-specific skill metrics
- Career-wide driver profiling
- Automated hyperparameter tuning

### [v1.4] – Foundations
- Historical database (2001–present)
- Random Forest ML pipeline
- Full-season backtesting
- Core accuracy metrics (Top-K, MAE)

---

## 🔮 Next Version

**v2.0 – Probabilistic & Ranking Models**
- Qualifying as Learning-to-Rank
- Gradient Boosting (LightGBM / CatBoost)
- Probabilistic race outcome distributions
