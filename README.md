# 🔋 Lithium-Ion Battery Analytics & Cycle Life Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4223?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **A comprehensive Data Science project analyzing high-frequency Li-ion battery sensor data to derive operational insights and develop Deep Learning models for Cycle Number estimation.**

---

## 📖 Project Overview
This project focuses on the analysis and predictive modeling of Lithium-Ion battery behavior using a dataset of over **680,000** high-frequency measurement records. The primary goal was to move beyond simple data visualization to build robust Machine Learning pipelines capable of estimating the **Cycle Number** (an indicator of battery age/health) based on instantaneous electrical parameters.

The project implements a full end-to-end pipeline: **Data Cleaning -> Feature Engineering -> EDA -> Model Training (Linear vs. Tree vs. Deep Learning) -> Evaluation**.

## 🎯 Objectives
* **Operational Analytics**: Extract discrete charging/discharging sessions and analyze voltage/current distributions.
* **Feature Engineering**: Create physics-informed features (Power, Rolling Statistics, Derivatives) to capture temporal dynamics.
* **Predictive Modeling**: Compare traditional ML models (Random Forest, XGBoost) against Deep Learning (MLP Regressor) to predict battery cycle life.
* **Interpretability**: Use Permutation Importance to understand which physical parameters drive battery aging predictions.

---

## 📊 Dataset Description
The dataset consists of time-series sensor data from a Lithium-Ion battery.
* **Size**: 687,212 entries.
* **Key Features**:
    * `Voltage (mV)`: Instantaneous terminal voltage.
    * `Current (mA)`: Instantaneous current (Positive = Charge, Negative = Discharge).
    * `Capacity (mAh)`: Accumulated capacity.
    * `Status`: Operational label (e.g., "Con-C Discharge").
* **Target Variable**: `Cycles` (Cycle Number).

---

## ⚙️ Methodology

### 1. Data Preprocessing & Cleaning
* **Mode Derivation**: Physically derived operational modes (`Charging`, `Discharging`, `Idle`) based on Current thresholds (+/- 10mA).
* **Session Identification**: Grouped continuous time-steps into unique sessions (202 total sessions identified) to analyze duration and stability.
* **Imputation**: Handled missing values using Forward/Backward filling to preserve time-series continuity.

### 2. Advanced Feature Engineering
To capture the battery's "memory" and physical state, I engineered 12+ new features:
* **Physics-Based**: `Power` ($V \times I$), `VoltxCap` ($V \times Capacity$).
* **Temporal Dynamics**: Rolling Means and Standard Deviations (Window sizes: 50, 200) for Voltage and Current.
* **Derivatives**: `dV` (Delta Voltage) and `dI` (Delta Current) to capture rates of change.
* **Scaling**: Applied `StandardScaler` to normalize features, which was critical for the convergence of the Neural Network model.

### 3. Exploratory Data Analysis (EDA)
* **Voltage Skewness (-0.65)**: Identified a negative skew indicating the battery spends most time at high States of Charge (SoC).
* **Current Multimodality**: Confirmed distinct operating states with a dominant peak at **-2500mA** (Constant Current Discharge).
* **Capacity Fade**: Visualized the degradation of maximum capacity over 100+ cycles, confirming the physical aging process.

---

## 🚀 Model Development & Performance
We evaluated 5 different regression models using **Time-Series Cross-Validation** to strictly separate past training data from future test data.

| Model | MAE (Cycles) | Training Time (s) | Key Insight |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 50.38 | 0.13s | Baseline; too simple for complex aging dynamics. |
| **Decision Tree** | 43.35 | 22.04s | Captured basic non-linear thresholds. |
| **Random Forest** | 40.97 | 312.01s | Stable but computationally expensive. |
| **XGBoost** | 39.15 | **13.08s** | **Efficiency Champion**; fast and accurate. |
| **MLP Regressor (Neural Net)** | **30.23** | 814.20s | **Performance Champion**; captured complex continuous functions. |

### 🏆 Best Model: MLP Regressor
The **Multi-Layer Perceptron (MLP)** significantly outperformed tree-based models.
* **Architecture**: Hidden Layers (64, 32), ReLU activation.
* **MAE**: **30.23** (The prediction is off by only ~30 cycles on average).
* **Why it won**: Battery aging involves high-dimensional, continuous interactions (electrochemical states) that are better approximated by deep learning layers than the hard decision boundaries of Decision Trees.

---

## 💡 Key Insights & Findings

### 1. The "State Aliasing" Problem
Despite low MAE, all models showed negative $R^2$ scores. This highlights a fundamental challenge in battery analytics: **State Aliasing**. A voltage of 3.8V at Cycle 5 looks nearly identical to 3.8V at Cycle 100. Without deep historical context (lifetime cumulative energy), models struggle to distinguish identical instantaneous states, even if they minimize absolute error well.

### 2. Feature Importance (Physics vs. Raw Data)
Using Permutation Importance, we discovered:
* **Top Predictors**: `Power` and `VoltxCap` (Interaction features) were orders of magnitude more important ($10^{14}$) than raw sensor data.
* **Noise**: Raw `Current` had negative importance, likely because the dataset was dominated by Constant Current (CC) discharging, making the raw value a static, non-predictive feature.

### 3. Deep Learning vs. Traditional ML
While sophisticated ensembles like XGBoost plateaued at an MAE of ~39, the Neural Network broke through this barrier. This suggests that for battery health estimation, **Deep Learning offers a tangible accuracy advantage** worth the extra training cost.

---

## 🛠️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/A5HW1NRA7AN/EMO.energy.git](https://github.com/A5HW1NRA7AN/EMO.energy.git)
    cd EMO.energy
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

3.  **Run the Notebook**:
    Open `assignment.ipynb` in Jupyter Notebook or VS Code to reproduce the analysis.

---

## 👤 Author
**Ashwin Rajan**
* **Role**: AI/ML Engineer & Researcher
* **Focus**: Deep Learning, Predictive Analytics, and Computer Vision.
* **Contact**: [LinkedIn](https://www.linkedin.com/in/ashwin-rajan-ai-ml-dev/) | [Email](ashwinrajanblr27@gmail.com)

---
*This project was completed as part of the Technical Assessment for the AI/ML Engineer Internship at EMO Energy.*