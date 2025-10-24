---
title: Data Overview
description: Structure and usage guidelines for datasets used in predictive maintenance experiments.
---

# Data Overview

This project uses the data/ directory for storing datasets used in experiments and documentation examples.

- Location: data/
- Contents: Raw CSVs, synthetic datasets, and experiment artifacts.
- Expected files (examples):
  - sensor_readings.csv — time-series sensor data (timestamp, sensor_id, value, unit)
  - maintenance_logs.csv — historical maintenance records (work_order_id, asset_id, date, failure_mode)
  - root_cause_labeled.csv — labeled dataset for root cause prediction (features..., root_cause)
  - synthetic_*.csv — generated data for demos and tests

## How data is used

- Ingestion: data_ingestion.py reads CSV files and performs basic preprocessing (imputation, scaling).
- Modeling:
  - predictive_failure_detection.py consumes time-series data for forecasting and anomaly scoring.
  - root_cause_prediction.py trains classification models on labeled historical data.
  - anomaly_detection.py computes anomaly scores on streaming or batch sensor inputs.

## Notes

- If you add new datasets, keep naming consistent and include a short README.txt in data/ with a description and schema.
- Large files should be stored outside version control or in object storage and referenced in documentation to keep the repo lightweight.