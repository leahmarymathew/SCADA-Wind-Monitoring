# SCADA-Based Wind Turbine Monitoring System

## Overview
This project implements a predictive maintenance system for wind turbines using SCADA data.
At-risk turbine states are detected using power-curve deviation between actual and theoretical output.

## Dataset
SCADA measurements include wind speed, actual power, theoretical power, and wind direction.

## Methodology
- Feature engineering using power deviation
- XGBoost classifier optimized for recall
- SHAP-based explainability for operational insights

## Results
- High recall on at-risk turbine detection
- Explainable power inefficiency indicators

## How to Run
```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train_model.py
python src/explain_shap.py
