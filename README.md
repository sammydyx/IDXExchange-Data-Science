# IDXExchange-Data-Science
IDX Exchange Data Science Project Work (Team: ds55)

# California Property Close Price Prediction

## Overview

This project aims to predict the final sale price of single-family residential properties in California using historical CRMLS transaction data. The dataset contains information on property characteristics, geographic location, school districts, taxes, fees, amenities, and transaction history.

The primary objective is to build and evaluate machine learning models that can accurately estimate a property's market value based on its attributes. The project follows a complete data science workflow, including data exploration, preprocessing, feature engineering, model development, hyperparameter tuning, and performance evaluation.

To avoid data leakage, listing price variables such as `ListPrice` and `OriginalListPrice` are excluded from the modeling process. The analysis focuses on non-leaky property and location features, including living area, bedrooms, bathrooms, lot size, property age, geographic information, school districts, and property amenities.

Several machine learning models are evaluated, including:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost / LightGBM

Model performance is assessed using metrics such as R², MAPE, and MdAPE. The final deliverable includes a trained prediction model, project documentation, and a presentation summarizing key findings and model performance.

## Project Goal

Develop a reliable property valuation model that can estimate the final sale price of a California single-family home using historical market data and property characteristics.
