"""Improved time-aware LightGBM experiment for California close prices.

Run from the repository root:
    python final_improvement/07_improved_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 420
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DATA_FILES = [
    ROOT / f"CRMLSSold{year}{month:02d}.csv"
    for year, months in ((2025, range(5, 13)), (2026, range(1, 7)))
    for month in months
]

NUMERIC_COLUMNS = [
    "Latitude",
    "Longitude",
    "LivingArea",
    "ParkingTotal",
    "YearBuilt",
    "BathroomsTotalInteger",
    "BedroomsTotal",
    "Stories",
    "MainLevelBedrooms",
    "GarageSpaces",
    "AssociationFee",
    "LotSizeSquareFeet",
    "BuildingAreaTotal",
]

BINARY_COLUMNS = [
    "ViewYN",
    "PoolPrivateYN",
    "AttachedGarageYN",
    "FireplaceYN",
    "NewConstructionYN",
    "WaterfrontYN",
    "BasementYN",
]

CATEGORICAL_COLUMNS = [
    "MLSAreaMajor",
    "PostalCode",
    "City",
    "CountyOrParish",
    "HighSchoolDistrict",
    "Levels",
    "Flooring",
    "SubdivisionName",
    "AssociationFeeFrequency",
    "ElementaryDistrict",
    "HighDistrict",
    "UnifiedDistrict",
]

DATE_COLUMNS = [
    "CloseDate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-district-join",
        action="store_true",
        help="Skip the school-boundary spatial join for a faster diagnostic run.",
    )
    return parser.parse_args()


def load_sales() -> pd.DataFrame:
    missing = [str(path) for path in DATA_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    requested = set(
        NUMERIC_COLUMNS
        + BINARY_COLUMNS
        + CATEGORICAL_COLUMNS[:9]
        + DATE_COLUMNS
        + ["PropertyType", "PropertySubType", "ClosePrice"]
    )

    frames = []
    for path in DATA_FILES:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in header if column in requested]
        monthly = pd.read_csv(path, usecols=usecols, low_memory=False)
        monthly["SourceFile"] = path.name
        frames.append(monthly)

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[
        df["PropertyType"].eq("Residential")
        & df["PropertySubType"].eq("SingleFamilyResidence")
    ].copy()
    return df


def add_school_districts(df: pd.DataFrame) -> pd.DataFrame:
    shapefile = ROOT / "week6/DistrictAreas2425/DistrictAreas2425.shp"
    if not shapefile.exists():
        raise FileNotFoundError(f"School district shapefile not found: {shapefile}")

    districts = gpd.read_file(shapefile)[
        ["DistrictNa", "DistrictTy", "geometry"]
    ].to_crs("EPSG:4326")

    valid = df["Latitude"].between(32, 42) & df["Longitude"].between(-125, -114)
    points = gpd.GeoDataFrame(
        df.loc[valid, ["Latitude", "Longitude"]].copy(),
        geometry=gpd.points_from_xy(
            df.loc[valid, "Longitude"], df.loc[valid, "Latitude"]
        ),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, districts, how="left", predicate="within")
    district_features = joined.reset_index().pivot_table(
        index="index",
        columns="DistrictTy",
        values="DistrictNa",
        aggfunc="first",
    )
    district_features = district_features.rename(
        columns={
            "Elementary": "ElementaryDistrict",
            "High": "HighDistrict",
            "Unified": "UnifiedDistrict",
        }
    )
    return df.join(
        district_features.reindex(
            columns=["ElementaryDistrict", "HighDistrict", "UnifiedDistrict"]
        )
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df = df.loc[df["ClosePrice"].gt(0) & df["CloseDate"].notna()].copy()

    for column in NUMERIC_COLUMNS:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    truthy = {"true", "t", "yes", "y", "1"}
    falsy = {"false", "f", "no", "n", "0"}
    for column in BINARY_COLUMNS:
        if column not in df:
            df[column] = np.nan
        normalized = df[column].astype("string").str.strip().str.lower()
        df[column] = normalized.map(
            lambda value: 1.0 if value in truthy else (0.0 if value in falsy else np.nan)
        )

    df["CloseYear"] = df["CloseDate"].dt.year
    df["CloseMonth"] = df["CloseDate"].dt.month
    df["CloseQuarter"] = df["CloseDate"].dt.quarter
    df["SaleMonthIndex"] = (
        (df["CloseYear"] - df["CloseYear"].min()) * 12 + df["CloseMonth"]
    )
    df["PropertyAgeAtSale"] = (df["CloseYear"] - df["YearBuilt"]).clip(0, 300)
    df["HasHOA"] = df["AssociationFee"].fillna(0).gt(0).astype(float)
    df["HasGarage"] = df["GarageSpaces"].fillna(0).gt(0).astype(float)
    df["LivingAreaPerBedroom"] = np.where(
        df["BedroomsTotal"].gt(0), df["LivingArea"] / df["BedroomsTotal"], np.nan
    )
    df["BathroomsPerBedroom"] = np.where(
        df["BedroomsTotal"].gt(0),
        df["BathroomsTotalInteger"] / df["BedroomsTotal"],
        np.nan,
    )
    df["LivingToLotRatio"] = np.where(
        df["LotSizeSquareFeet"].gt(0),
        df["LivingArea"] / df["LotSizeSquareFeet"],
        np.nan,
    )
    df["LogLivingArea"] = np.log1p(df["LivingArea"].clip(lower=0))
    df["LogLotSizeSquareFeet"] = np.log1p(df["LotSizeSquareFeet"].clip(lower=0))

    return df.sort_values("CloseDate").reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["CloseDate"] < "2026-05-01"].copy()
    validation = df.loc[df["CloseDate"].between("2026-05-01", "2026-05-31")].copy()
    test = df.loc[df["CloseDate"].between("2026-06-01", "2026-06-30")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("The time-based train/validation/test split produced an empty set.")
    return train, validation, test


def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    engineered_numeric = [
        "CloseYear",
        "CloseMonth",
        "CloseQuarter",
        "SaleMonthIndex",
        "PropertyAgeAtSale",
        "HasHOA",
        "HasGarage",
        "LivingAreaPerBedroom",
        "BathroomsPerBedroom",
        "LivingToLotRatio",
        "LogLivingArea",
        "LogLotSizeSquareFeet",
    ]
    numeric = [column for column in NUMERIC_COLUMNS + BINARY_COLUMNS + engineered_numeric if column in train]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in train]

    train_x = train[numeric + categorical].copy()
    validation_x = validation[numeric + categorical].copy()
    test_x = test[numeric + categorical].copy()

    for column in numeric:
        median = train_x[column].replace([np.inf, -np.inf], np.nan).median()
        for frame in (train_x, validation_x, test_x):
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan).fillna(median)

    for column in categorical:
        train_values = train_x[column].astype("string").fillna("__MISSING__")
        categories = sorted(train_values.unique().tolist())
        for frame in (train_x, validation_x, test_x):
            values = frame[column].astype("string").fillna("__MISSING__")
            values = values.where(values.isin(categories), "__MISSING__")
            frame[column] = pd.Categorical(values, categories=categories)

    return train_x, validation_x, test_x, categorical


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 1, None)
    ape = np.abs((y_true - prediction) / y_true)
    return {
        "R2": r2_score(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "MAE": mean_absolute_error(y_true, prediction),
        "MAPE": ape.mean() * 100,
        "MdAPE": np.median(ape) * 100,
        "RMSLE": np.sqrt(
            np.mean((np.log1p(y_true) - np.log1p(prediction)) ** 2)
        ),
    }


def fit_model(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    validation_y: np.ndarray,
    categorical: list[str],
    log_target: bool,
) -> tuple[lgb.LGBMRegressor, int]:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=10,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_y = np.log1p(train_y) if log_target else train_y
    eval_y = np.log1p(validation_y) if log_target else validation_y
    model.fit(
        train_x,
        fit_y,
        categorical_feature=categorical,
        eval_X=validation_x,
        eval_y=eval_y,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model, model.best_iteration_


def refit_model(
    development_x: pd.DataFrame,
    development_y: np.ndarray,
    categorical: list[str],
    log_target: bool,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=10,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_y = np.log1p(development_y) if log_target else development_y
    model.fit(development_x, fit_y, categorical_feature=categorical)
    return model


def predict(model: lgb.LGBMRegressor, x: pd.DataFrame, log_target: bool) -> np.ndarray:
    prediction = model.predict(x, num_iteration=model.best_iteration_)
    return np.expm1(prediction) if log_target else prediction


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sales()
    if args.skip_district_join:
        for column in ("ElementaryDistrict", "HighDistrict", "UnifiedDistrict"):
            df[column] = "__SKIPPED__"
    else:
        df = add_school_districts(df)
    df = engineer_features(df)
    train, validation, test = split_data(df)
    train_x, validation_x, test_x, categorical = prepare_features(
        train, validation, test
    )
    train_y = train["ClosePrice"].to_numpy(float)
    validation_y = validation["ClosePrice"].to_numpy(float)
    test_y = test["ClosePrice"].to_numpy(float)

    rows = []
    models = {}
    for label, use_log in (("Raw target", False), ("Log1p target", True)):
        selection_model, best_iteration = fit_model(
            train_x,
            train_y,
            validation_x,
            validation_y,
            categorical,
            use_log,
        )
        validation_prediction = predict(selection_model, validation_x, use_log)
        rows.append(
            {
                "Model": label,
                "Split": "Validation (2026-05)",
                "Rows": len(validation_y),
                "BestIteration": best_iteration,
                **metrics(validation_y, validation_prediction),
            }
        )

        development_x = pd.concat([train_x, validation_x], ignore_index=True)
        development_y = np.concatenate([train_y, validation_y])
        final_model = refit_model(
            development_x,
            development_y,
            categorical,
            use_log,
            best_iteration,
        )
        models[label] = final_model
        test_prediction = predict(final_model, test_x, use_log)
        rows.append(
            {
                "Model": label,
                "Split": "Test (2026-06, refit through May)",
                "Rows": len(test_y),
                "BestIteration": best_iteration,
                **metrics(test_y, test_prediction),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    log_model = models["Log1p target"]
    test_prediction = predict(log_model, test_x, True)
    predictions = pd.DataFrame(
        {
            "CloseDate": test["CloseDate"].to_numpy(),
            "Actual": test_y,
            "Predicted": test_prediction,
            "Error": test_prediction - test_y,
            "AbsoluteError": np.abs(test_prediction - test_y),
            "AbsolutePercentageError": np.abs(test_prediction - test_y) / test_y * 100,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "log_lightgbm_test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "Feature": train_x.columns,
            "Importance": log_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "log_lightgbm_feature_importance.csv", index=False)

    metadata = {
        "data_files": [path.name for path in DATA_FILES],
        "train_period": [str(train.CloseDate.min().date()), str(train.CloseDate.max().date())],
        "validation_period": [
            str(validation.CloseDate.min().date()),
            str(validation.CloseDate.max().date()),
        ],
        "test_period": [str(test.CloseDate.min().date()), str(test.CloseDate.max().date())],
        "features": train_x.columns.tolist(),
        "categorical_features": categorical,
        "district_join_skipped": args.skip_district_join,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(results.to_string(index=False))
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
