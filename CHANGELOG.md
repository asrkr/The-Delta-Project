# 📜 Changelog

All notable changes to **The Delta Project** are documented in this file.

The project follows a versioned, iterative development approach focused on model stability, domain intelligence, and progressive realism.

---

## [v1.9.0] – Interactive Dashboard  
**Release date:** 2026-09-24

### ✨ Added
- **Streamlit dashboard (`app.py`, 487 lignes)** — interface F1-themed en 3 onglets :
  - **🔮 Prédiction** — Oracle (grille prédite) vs Analyst (grille réelle), podium stylé + table des résultats.
  - **🔄 Données** — refresh incrémental Ergast / calendrier / FastF1 / sprints / dernière qualif avec logs live.
  - **🛠️ Mode Dev** — backtest walk-forward saison complète (Winner / Top 3 / Top 5 / Top 10 / MAE) via `dev_tools/simulateur_saison`.
- **`src/config.py` (single source of truth)** :
  - Centralise `QUALIF_PARAMS` (LightGBM Ranker, tuned Optuna), `RACE_PARAMS` (RandomForest) et `QUALIF_FEATURES` / `RACE_FEATURES`.
  - Supprime la duplication entre `ml_model.train_models` et `QualifRankerLGBM.__init__`.
- **`requirements.txt`** — dépendances pinnées : `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `requests`, `fastf1`, `streamlit`, `optuna`.
- **`train_and_predict()` retourne désormais `pd.DataFrame`** des résultats classés (réutilisable par la GUI), en plus de l'affichage CLI.

### 🔄 Changed
- **`src/ml_model.py` refactor** :
  - `train_models()` lit hyperparams & feature lists depuis `src/config.py` (copy via `dict()` pour éviter la mutation).
  - `predict_race_outcome()` passe d'une boucle `predict()` par pilote à une **prédiction batchée** (une seule matrice `X_r` → un seul `model_race.predict()`), numériquement identique mais plus rapide.
  - `DEFAULT_PRED_GRID = 10` extrait en constante (cas rookies sans sortie du ranker).
  - `except:` nu → `except Exception:` + log `⚠️ Erreur prédiction course` et `return pd.DataFrame()` sur échec global.
- **`src/models/qualif_ranker.py`** importe `QUALIF_PARAMS` depuis `src/config.py` (défauts copiés, plus de duplication inline).
- **`README.md`** : badge `V1.9`, tech stack `lightgbm`/`streamlit`/`optuna`, arborescence à jour, roadmap Phase 4.5, section GUI `3-bis`.
- **`.gitignore`** : ajout `.streamlit/` (cache local Streamlit).

### 🐞 Fixed
- `src/data_manager.py:427` — `def extract_fastf1_features(...) -> None  :` → `-> None:` (double espace supprimé, lint).

### ✅ Validated
- Parité numérique vérifiée : prédiction batchée vs boucle pilote-par-pilote identique.
- Aucune fuite temporelle introduite (features toujours past-only).
- GUI testée : prédiction Oracle/Analyst, refresh données, backtest dev avec callback de progression.

### ⚠️ Design Notes
- La GUI réutilise le moteur existant sans modifier la logique métier ; `dev_tools/` reste non packagé (outil interne).
- Performances V1.8 inchangées (modèles identiques, seuls l'orchestration et l'UX évoluent).

---

## [v1.8.1] – Windows Encoding Fix  
**Release date:** 2026-09-24

### 🐞 Fixed
- **Crash emoji sur Windows (`main.py`)** — `UnicodeEncodeError` sur consoles `cp1252` lors de l'affichage `🏁`/`⚠️` avant toute prédiction.
  - Reconfiguration `sys.stdout`/`sys.stderr` en UTF-8 au démarrage via `reconfigure(encoding="utf-8")`, encapsulée en `try/except` (no-op si indisponible).

---

## [v1.8.0] – Clean Air & Weather Context  
**Release date:** 2026

### ✨ Added
- **Clean-air pace integration (FastF1)**:
  - New telemetry signal: `clean_air_pace`
  - Aggregated career feature: `career_clean_air_pace` (past-only, expanding mean with shift)
- **Deterministic race-day weather context**:
  - `is_rainy` (binary race context flag)
  - `track_temp` (track temperature context)
- **Wet-skill driver profiling (contextual, past-only)**:
  - `career_wet_skill` based on average *rainy* gain/loss (grid → finish) computed using strict past-only history
- **Optuna-only walk-forward tuning workflow (internal dev tool)**:
  - Qualifying and Race models optimized independently with time-split folds (train on past → validate on next race)
  - Defaults chosen to avoid manual fold/trial tuning and reduce overfitting to a single season
- **Extended validation battery**:
  - Stress-tested across tricky seasons (calendar irregularities, mixed formats, regulation shifts)
  - Forward-style validation including Oracle vs Analyst comparison

### 🔄 Changed
- **Race feature set enriched** with contextual signals (clean air + weather) while keeping the core “Dual Brain” logic unchanged.
- **Telemetry merge robustness improved**:
  - Safe defaults for missing FastF1 values (especially pre-2018 coverage gaps).
  - Type-safe filling/casting for `is_rainy`, `track_temp`, pit metrics.

### 🐞 Fixed
- Edge cases where missing external telemetry columns could silently degrade feature availability.
- Improved stability of “context” joins (weather / telemetry) so that prediction doesn’t depend on perfect coverage.

### ✅ Validated
- No detectable temporal leakage introduced by the new contextual features (past-only aggregation preserved).
- Analyst mode remains consistently strong (race model isolated with real grid).
- Oracle mode behaves realistically in forward conditions (controlled degradation when predicting both grid and race).

### ⚠️ Design Notes
- Clean-air / weather context is **deterministic** (not a stochastic simulator).
- FastF1-derived features are primarily meaningful from **2018+**; earlier seasons fall back to neutral/median defaults by design.
- Random race chaos remains intentionally out of scope (DNFs, safety cars, crashes, pure randomness).

---

## [v1.7.1] – Stability & Temporal Integrity  
**Release date:** 2026

### ✨ Added
- **Strict temporal integrity guarantees** across:
  - Single-race prediction pipeline (no future races used to compute features).
  - Internal season benchmarking workflow (walk-forward logic).
  - Hyperparameter tuning (race-by-race validation aligned with real usage).
- **Canonical driver identity layer**:
  - Introduced explicit aliasing to enforce consistent `DriverKey` across data sources
    (e.g. multi-first-name vs short-first-name discrepancies).
  - Prevents silent merge failures between Ergast / FastF1 / Sprint datasets.
- **Walk-forward tuning script (v1.7.1)**:
  - Hyperparameters optimized independently for Qualifying and Race models using
    strict time-based folds (train on past → validate on the next race).

### 🔄 Changed
- **Validation methodology upgraded**:
  - All evaluation workflows are now aligned around a single principle:
    *features must be computed using past-only information relative to the target race*.
- **Benchmark transparency**:
  - Internal backtesting is explicitly treated as a development/validation tool
    (not part of the public repository), while ensuring results remain comparable and reproducible.

### 🐞 Fixed
- **Temporal leakage sources** affecting:
  - Circuit impact estimation and global imputations when computed on full datasets.
  - Feature computation order in prediction/tuning workflows.
- **Driver identity inconsistencies** (e.g. differing given names across APIs) causing:
  - Missing merges,
  - Rookie edge cases,
  - Unstable feature availability for some drivers.

### ✅ Validated
- Training always uses **strictly prior races** for each target event.
- Canonical `DriverKey` prevents cross-source mismatches and improves merge stability.
- Tuning results are now directly applicable to real-world prediction usage.

### ⚠️ Design Notes
- This is a **stability release**: no new modelling concepts are introduced.
- Stochastic race events remain intentionally out of scope (DNFs, SC, crashes, randomness).
- Weather is still treated as non-stochastic in this branch; deterministic weather context is planned for v1.8.

---

## [v1.7.0] – The Sprint Update  
**Release date:** 2025

### ✨ Added
- **Sprint weekend contextual integration** (Ergast API):
  - Dedicated Sprint results dataset (`f1_sprint_results.csv`).
  - Safe, incremental loading without polluting main race results.
- **Sprint-aware race context features**:
  - `has_sprint` – explicit weekend format flag.
  - `sprint_pos` – Sprint finishing position.
  - `sprint_delta` – relative gain/loss during Sprint (grid → finish).
- **Additive feature engineering strategy**:
  - Sprint data enriches race context without modifying existing form or grid logic.
  - Non-sprint weekends handled via explicit gating (no implicit NaNs).
- **Sprint-aware benchmarking tools**:
  - Separate evaluation for Sprint vs Non-Sprint weekends.
  - Baseline comparison against real grid MAE.

### 🔄 Changed
- **Data loading pipeline extended** to merge Sprint data via temporal-safe joins.
- **Race model context enriched**, without altering qualifying or race targets.
- **Benchmark methodology refined**:
  - Explicit comparison against grid baseline.
  - Per-weekend format performance breakdown.

### ✅ Validated
- No regression on non-sprint weekends.
- Measurable MAE improvement on Sprint weekends (validated on 2021 season).
- Stable global performance across full seasons.

### ⚠️ Design Notes
- Sprint races are treated strictly as **contextual signals**, not prediction targets.
- Core RandomForest architecture remains unchanged.
- This version finalizes the data foundation ahead of model migration.

---

## [v1.6.0] – Telemetry Integration  
**Release date:** 2025

### ✨ Added
- **FastF1 telemetry integration**:
  - Average race pace.
  - Best lap time.
  - Pit stop loss estimation.
- **Race execution awareness** through telemetry-derived features.
- **Real grid injection**:
  - Ability to run race predictions using real qualifying results.
  - Enables clear separation between qualifying accuracy and race modeling quality.
- **Season-level pace normalization** (`pace_rank_season`).
- **Improved driver identity handling**:
  - Stable `DriverKey` generation to avoid name collisions (e.g. Verstappen, Schumacher).

### 🔄 Changed
- **Race Model feature set rebalanced**:
  - Grid-related contextual features were evaluated and simplified for stability.
  - Telemetry features prioritized over over-engineered grid transformations.
- **Qualifying ↔ Race decoupling reinforced**:
  - Qualifying predicts grid only.
  - Race model focuses on execution and pace conditional on starting position.
- **Hyperparameter tuning pipeline updated** to reflect the new feature space.
- **Model benchmarks redefined** using two explicit scenarios:
  - *Oracle Mode* (predicted grid).
  - *Analyst Mode* (real grid).

### 🗑️ Removed
- Experimental grid normalization variants that degraded generalization:
  - `grid_z`.
  - `grid_percent`.
- Overly synthetic grid deltas that did not improve Oracle Mode performance.

### 🐞 Fixed
- Driver name / team mismatches causing label encoder crashes.
- Rookie / mid-season team edge cases.
- Silent data leakage between training and prediction phases.

### ⚠️ Known Limitations
- Oracle Mode (full AI prediction) remains limited by qualifying model accuracy.
- Stochastic race events are intentionally not modeled:
  - DNFs.
  - Safety Cars.
  - Weather randomness.
- Sprint formats not yet supported.

---

## Previous Versions

### [v1.5] – Domain Intelligence
- Recent form (rolling averages).
- Circuit-specific skill metrics.
- Career-wide driver profiling.
- Automated hyperparameter tuning.

### [v1.4] – Foundations
- Historical database (2001–present).
- Random Forest ML pipeline.
- Full-season backtesting.
- Core accuracy metrics (Top-K, MAE).

---

## 🔮 Next Version

**v2.0 – Probabilistic & Ranking Models**
- ~~Qualifying as Learning-to-Rank~~ → done in v1.9 (LightGBM Ranker).
- Gradient Boosting refinements (CatBoost, deeper Optuna search).
- Probabilistic race outcome distributions.
