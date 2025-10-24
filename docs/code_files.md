---
title: Source Code Overview
description: Summary of core Python modules used in the AI-driven predictive maintenance system with links to source.
---

# Source Code Overview

Below are the core Python modules in the src/ directory with brief descriptions and links to the source code in the repository.

- anomaly_detection.py
  - Purpose: Implements anomaly detection using Isolation Forest and Autoencoders for sensor streams.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/anomaly_detection.py
  - Local path: src/anomaly_detection.py

- data_ingestion.py
  - Purpose: CSV-based ingestion and preprocessing (missing values, scaling) for structured datasets.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/data_ingestion.py
  - Local path: src/data_ingestion.py

- dynamic_resource_allocation.py
  - Purpose: Reinforcement learning and optimization utilities for dynamic resource allocation based on predictions and RUL.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/dynamic_resource_allocation.py
  - Local path: src/dynamic_resource_allocation.py

- generate_root_cause_data.py
  - Purpose: Synthetic data generation tailored to root cause classification experiments.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/generate_root_cause_data.py
  - Local path: src/generate_root_cause_data.py

- generate_synthetic_data.py
  - Purpose: General synthetic dataset utilities for time-series forecasting and anomaly detection.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/generate_synthetic_data.py
  - Local path: src/generate_synthetic_data.py

- predictive_failure_detection.py
  - Purpose: Predictive failure detection pipeline combining time-series forecasting and anomaly scoring.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/predictive_failure_detection.py
  - Local path: src/predictive_failure_detection.py

- root_cause_prediction.py
  - Purpose: Root cause classification using interpretable models (e.g., Decision Trees) with label encoding.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/root_cause_prediction.py
  - Local path: src/root_cause_prediction.py

- time_series_forecasting.py
  - Purpose: LSTM/GRU-based forecasting for key metrics; forms the basis for proactive anomaly detection.
  - GitHub: https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/blob/main/src/time_series_forecasting.py
  - Local path: src/time_series_forecasting.py