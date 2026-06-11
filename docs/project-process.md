# Project Process

## 1. Data Loading
The notebook loads stock monthly data, index returns, company profile attributes, training targets, testing IDs, and optional macroeconomic series.

## 2. Data Understanding and Quality Checks
Key checks include dataframe shape, data types, `stock_id` and `month_id` regex validation, missing values, duplicated keys, and categorical consistency.

## 3. Exploratory Analysis
The analysis explores OHLC stock patterns, outliers, numeric distributions, and correlations with `excess_return`.

## 4. Feature Engineering
The workflow uses price/volume features, return and volatility indicators, categorical company descriptors, scaling, log transforms, one-hot encoding, and lagged features for forecasting.

## 5. Classification Task
The target is `outperform_binary`, indicating whether the stock outperformed the index. Models compared include Logistic Regression, Random Forest, SVM, KNN, and Naive Bayes.

## 6. Regression Task
The target is `excess_return`. Linear, regularized, support-vector, and tree-based regressors are compared using RMSE, including TimeSeriesSplit validation.

## 7. Inference Output
The notebook prepares July 2023 inference logic and writes regression predictions to `testing_targets_regression.csv` when data is available.
