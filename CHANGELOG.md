# 📜 Changelog – The Delta Project

This document tracks the evolution of the model’s logic and capabilities.
It focuses on **what intelligence was added** at each stage, not implementation details.

---

## [V1.6] – Strategy-aware & Robust Pipeline

### Summary
V1.6 focuses on **race execution realism and robustness**.
The model moves beyond pure grid/finish relationships and integrates race-pace context while preserving stability.

### Key changes
- Integration of **FastF1 race-pace features** (average pace, best lap, consistency).
- Introduction of a **robust pit loss estimation**, used as a career-level signal rather than event noise.
- Improved handling of missing or partial FastF1 data (graceful fallbacks).
- Unified feature pipeline between **production model and hyperparameter tuning**.
- Backend handling of **real qualifying grids** when available (no user intervention required).
- Significant stabilization of predictions across seasons and circuits.

### Impact
- Improved realism in race outcome predictions.
- Reduced volatility between similar drivers.
- Better discrimination between strategic strength and raw performance.

---

## [V1.5] – Domain Intelligence

### Summary
V1.5 introduced **F1-specific reasoning** into the model.
Predictions are no longer purely statistical, but context-aware.

### Key changes
- Driver **recent form** (rolling average over previous races).
- **Circuit importance** based on historical grid-to-finish correlation.
- Driver **career averages**, including circuit-specific skills.
- Walk-forward evaluation and automated **hyperparameter tuning**.
- Clear separation between training data and prediction race.

### Impact
- Major accuracy gains in winner and Top-3 predictions.
- Strong reduction in Mean Absolute Error (MAE).
- Model begins to reflect real F1 dynamics (track position sensitivity).

---

## [V1.4] – Foundations

### Summary
Core pipeline and dataset construction.

### Key changes
- Incremental Ergast scraping (2001–present).
- Clean historical race dataset.
- First Random Forest pipeline for qualifying and race predictions.
- Dynamic participant retrieval per race.
- Initial performance metrics (Top-K accuracy, MAE).

### Impact
- Stable base for all future iterations.
- Enabled rapid experimentation and feature iteration.
