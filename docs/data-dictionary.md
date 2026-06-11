# Data Dictionary

## Core Files

| File | Description | Key Columns |
|---|---|---|
| `stock_data.csv` | Monthly stock price, volume, return, and volatility features | `stock_id`, `month_id`, OHLC, volume, returns, volatility |
| `company_info.csv` | Static company descriptors | `stock_id`, sector, maturity, market cap category, profitability profile |
| `index.csv` | Monthly index-level return and value | `month_id`, `index_return`, `index_value` |
| `training_targets.csv` | Supervised learning labels | `stock_id`, `month_id`, `outperform_binary`, `excess_return` |
| `testing_targets.csv` | July 2023 stocks for inference | `stock_id`, `month_id` |

## Optional Macro Files

| File | Description |
|---|---|
| `fed funds rate.csv` | US policy rate series |
| `fed inflation rate.csv` | US inflation series |
| `fed unemployment rate.csv` | US unemployment series |
| `us 5 year treasury.csv` | 5-year treasury yield series |
| `us 10 year treasury.csv` | 10-year treasury yield series |
| `vix index.csv` | Market volatility proxy |

## Target Definitions

| Target | Type | Meaning |
|---|---|---|
| `outperform_binary` | Classification | 1 if stock outperforms the index, otherwise 0 |
| `excess_return` | Regression | Stock return minus index return |
