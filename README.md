# California Property Close Price Prediction

Team **ds55** data science project using California Regional Multiple Listing
Service (CRMLS) residential sales data.

## Project Overview

The goal of this project is to predict the final sale price (`ClosePrice`) of a
California single-family home from information that is available about the
property and its location. The repository follows the full modeling workflow:
exploratory analysis, preprocessing, baseline modeling, feature engineering,
gradient-boosting models, and held-out evaluation.

Listing-price fields such as `ListPrice` and `OriginalListPrice` are excluded
from model inputs because they would leak direct information about the final
sale price.

## Dataset

The source files are monthly exports of closed-sale records from **CRMLS**, made
available for this project through IDX Exchange. The current pipeline reads 14
monthly files covering **May 2025 through June 2026**. Only records satisfying
both of the following conditions are retained:

- `PropertyType == "Residential"`
- `PropertySubType == "SingleFamilyResidence"`

The CSV files are stored in the repository root and are not redistributed by
this README. Users must have authorized access to CRMLS/IDX Exchange data and
must preserve the filenames referenced in the notebooks.

## Preprocessing

The main preprocessing workflow is in
[`week3/version2/02_preprocessing.ipynb`](week3/version2/02_preprocessing.ipynb).
It performs the following steps:

1. Loads and combines the monthly CRMLS files.
2. Filters the data to residential single-family homes.
3. Converts invalid domain values to missing values and removes unusable
   records.
4. Removes rows with missing, zero, or negative `ClosePrice`.
5. Uses May 2025-May 2026 records for model development and reserves June 2026
   as the held-out test set.
6. Calculates target outlier thresholds from the training data only and removes
   extreme training targets without using the held-out test target distribution.
7. Drops leakage-prone, redundant, and non-modeling columns.
8. Imputes numeric values with statistics learned from the training set and
   fills categorical/indicator fields with explicit defaults.
9. One-hot encodes `MLSAreaMajor` and aligns the train/test feature columns.

Week 6 adds school-district features through a geographic point-in-polygon join
and engineers the following property features:

- HOA membership indicator
- Living area per bedroom
- Living-area-to-lot-size ratio
- Bathrooms per bedroom
- Garage-related features

The final feature set is saved as
`week6/data_versions/v6_all_features.csv.gz`.

## Models Tested

The project evaluates these regression models:

- Mean-value dummy regressor
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

Early models establish interpretable baselines. The tree ensembles capture the
nonlinear relationships and feature interactions present in housing data.
XGBoost and LightGBM are tuned on a development/validation split, then evaluated
once on the same June 2026 held-out test set.

## Best Results

The final test set contains **12,853 homes**. LightGBM produced the strongest
overall result.

| Model | Test R² | RMSE | MAE | MAPE | MdAPE |
|---|---:|---:|---:|---:|---:|
| LightGBM | **0.7650** | **$744,924** | **$236,661** | **23.86%** | **11.10%** |
| XGBoost | 0.7622 | $749,317 | $271,761 | 28.18% | 14.05% |
| Random Forest | 0.5034 | $1,182,835 | Not recorded | 19.69% | 9.65% |
| Linear Regression | 0.4943 | $1,193,534 | Not recorded | 31.13% | 18.01% |

Results are recorded in
[`week8/metrics_summary.csv`](week8/metrics_summary.csv) and
[`week8/price_band_summary.csv`](week8/price_band_summary.csv).

LightGBM also has the lowest MAE, MAPE, and MdAPE of the two final boosting
models. Errors remain much larger for the highest-priced homes: rare luxury
sales dominate RMSE and are the main limitation of the current model. Price-band
R² values should be interpreted cautiously because each band has a much narrower
target range than the full test set.

## Repository Structure

```text
.
|-- week2/01_exploration.ipynb          # Exploratory data analysis
|-- week3/version2/02_preprocessing.ipynb
|-- week4/03_baseline_model.ipynb       # Linear Regression baseline
|-- week5/04_model_comparison.ipynb     # Trees and Random Forest
|-- week6/new_features.ipynb            # Spatial and ratio features
|-- week6/data_versions/                # Feature-set snapshots
|-- week7/05_advanced_models.ipynb      # XGBoost and LightGBM
|-- week7/outputs/                       # Predictions, metrics, importance
|-- week8/06_evaluation.ipynb           # Overall and price-band evaluation
|-- week8/metrics_summary.csv
`-- week8/price_band_summary.csv
```

## Reproduce the Analysis

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn \
  geopandas xgboost lightgbm
```

### 2. Confirm the input data

Place the monthly CRMLS CSV files in the repository root. The notebooks expect
files named `CRMLSSold202505.csv` through `CRMLSSold202606.csv`. Keep the Week 6
school-district shapefile and its companion files together in
`week6/DistrictAreas2425/`.

### 3. Run the notebooks

Notebook paths are relative to each notebook's folder. Start Jupyter from the
repository root, open each notebook, and run all cells in this order:

1. `week2/01_exploration.ipynb`
2. `week3/version2/02_preprocessing.ipynb`
3. `week4/03_baseline_model.ipynb`
4. `week5/04_model_comparison.ipynb`
5. `week6/new_features.ipynb`
6. `week6/rerun_baseline_model.ipynb`
7. `week6/rerun_additional_model.ipynb`
8. `week7/05_advanced_models.ipynb`
9. `week8/06_evaluation.ipynb`

```bash
jupyter lab
```

If a notebook is executed non-interactively, set its working directory to the
folder containing that notebook so its relative paths continue to resolve.

## Launch the App

An application entry point and a serialized final model are **not currently
included in this repository**, so there is no valid app launch command yet.
Before deployment, export the fitted LightGBM model together with its exact
preprocessing schema and add the Week 9 application entry point (for example,
`week9/app.py`). Once those files exist, this section should be updated with the
tested installation and launch command rather than a placeholder command.

## Limitations

- The model is trained on CRMLS records and may not generalize to transactions
  outside the represented California markets or time period.
- Luxury homes are sparse and have substantially larger absolute errors.
- School-district assignments depend on coordinate quality and the supplied
  boundary vintage.
- This model is an academic estimate, not an appraisal or financial advice.
