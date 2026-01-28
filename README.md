[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/leahmarymathew/SCADA-Wind-Monitoring/blob/main/notebooks/SCADA_Wind_Turbine_Monitoring_Demo.ipynb
)


# SCADA-Based Wind Turbine Monitoring System
Predictive Maintenance using Machine Learning, Explainability & Parallel Computing

---

## Overview

This project implements a SCADA-based predictive maintenance system for wind turbines, designed to identify at-risk operating states using power-curve deviation analysis and machine learning.

The system integrates:
- Data-driven anomaly detection
- Explainable machine learning (SHAP)
- High-performance inference using OpenMP and MPI

The overall design mirrors real-world wind-farm monitoring pipelines, where large volumes of SCADA data must be processed efficiently and reliably.

---

## Problem Statement

Wind turbines continuously generate SCADA data describing their operational behavior. Deviations between actual power output and theoretical power curves are strong early indicators of turbine inefficiencies or faults.

These deviations may arise due to:
- Blade wear or aerodynamic inefficiency
- Yaw misalignment
- Generator or control-system degradation

The objective of this project is to detect at-risk turbine states early and provide interpretable insights to support operational and plant engineering decisions.

---

## Methodology

### SCADA Data Processing
- Cleaned and validated raw SCADA measurements
- Removed physically invalid operating points
- Standardized feature naming and formatting

### Power-Curve Deviation Modeling
- Computed normalized deviation between theoretical and actual power output
- Used deviation under sufficient wind conditions as a risk indicator
- Generated realistic at-risk labels aligned with industry practices

### Machine Learning (XGBoost)
- Trained an XGBoost classifier optimized for recall
- Addressed class imbalance using weighted loss
- Evaluated performance on unseen SCADA samples

### Explainability (SHAP)
- Applied SHAP to explain individual and global predictions
- Identified dominant contributors to turbine risk
- Enabled transparent diagnostics for non-ML stakeholders

### High-Performance Inference
- OpenMP used for shared-memory parallel inference
- MPI used to simulate distributed turbine clusters
- Benchmarked inference latency and scalability

---

## System Architecture

```

SCADA CSV
|
|-- Preprocessing & Labeling
|
|-- Feature Engineering (Power Deviation)
|
|-- XGBoost Model
|      |-- Risk Prediction
|      |-- SHAP Explainability
|
|-- OpenMP Parallel Inference
|
|-- MPI Distributed Turbine Clusters

```

---

## Dataset

Input SCADA features:
- Wind Speed
- Actual Power Output
- Theoretical Power Output
- Wind Direction
- Timestamp

Derived feature:
- Power Deviation Ratio

The dataset structure reflects standard SCADA measurements used in operational wind farms.

---

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- XGBoost
- SHAP
- OpenMP (C)
- MPI
- Matplotlib
- VS Code

---

## Performance Highlights

- High recall for detecting at-risk turbine states
- Explainable predictions aligned with physical turbine behavior
- OpenMP inference achieved approximately 2.5× speedup with ~40% latency reduction
- MPI-based distributed inference showed scalable performance across multiple processes

Exact performance values depend on hardware configuration and dataset size.

---

## Project Structure

```

scada-wind-turbine-monitoring/
|
|-- data/
|   |-- raw_scada.csv
|   |-- processed_scada.csv
|
|-- src/
|   |-- preprocess.py
|   |-- train_model.py
|   |-- explain_shap.py
|   |-- openmp_inference.c
|   |-- mpi_inference.c
|
|-- benchmarks/
|   |-- inference_input.csv
|
|-- results/
|   |-- metrics.txt
|   |-- shap_summary.png
|   |-- openmp_benchmark.txt
|   |-- mpi_benchmark.txt
|
|-- Makefile
|-- requirements.txt
|-- README.md

```

---

## How to Run

Install dependencies:
```

pip install -r requirements.txt

```

Preprocess data:
```

python src/preprocess.py

```

Train model:
```

python src/train_model.py

```

Generate SHAP explanations:
```

python src/explain_shap.py

```

Run OpenMP inference benchmark:
```

make
OMP_NUM_THREADS=4 ./openmp_inference

```

Run MPI distributed inference:
```

mpirun -np 4 ./mpi_inference

```

---

## Engineering and Research Relevance

This project demonstrates:
- Industrial-style ML deployment on SCADA data
- Explainable AI for safety-critical systems
- Hybrid shared-memory and distributed-memory parallelism
- Scalable architecture suitable for large wind farms

The system is relevant for roles in industrial AI, renewable energy analytics, high-performance computing, and applied machine learning research.

---

## Future Work

- Real-time SCADA data streaming
- Hybrid OpenMP + MPI inference pipelines
- Model drift detection
- Fault-type classification
- Integration with operational dashboards

---

## License

This project is intended for educational and research purposes.

