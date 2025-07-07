# CNC Milling Performance Analysis and Fault Detection

## Project Overview

This project focuses on the analysis of CNC milling operations and the development of predictive models to detect tool wear, identify inadequate clamping, and predict process completion using time series sensor data collected from machining experiments.

---

## Skills Gained from This Project

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering for Neural Networks
* Building and Training Artificial Neural Networks (ANNs)
* Model Evaluation and Optimization
* Time Series Analysis and Feature Engineering
* Predictive Maintenance and Fault Detection
* Deploying Deep Learning Models using AWS and Streamlit
* Model Evaluation and Performance Metrics

---

## Problem Statement

To analyze CNC milling operations and build predictive models for:

* Tool wear detection
* Inadequate clamping detection
* Process completion prediction

All based on time series sensor data from machining experiments.

---

## Business Use Cases

* **Predictive Maintenance:** Identify tool wear and schedule replacements to minimize downtime.
  
* **Quality Assurance:** Detect insufficient clamping to avoid defective parts.
  
* **Process Optimization:** Optimize parameters to minimize cycle times and reduce costs.
  
* **Operational Safety:** Monitor parameters to ensure safe machining operations.

---

## Project Approach

1. **Data Understanding:**

   * Analyze `train.csv` and 18 time series files (`experiment_01.csv` to `experiment_18.csv`).

2. **Data Preprocessing:**

   * Handle missing values and anomalies in sensor readings.

3. **Feature Engineering:**

   * Extract statistical, temporal, and frequency-based features suitable for neural network models.

4. **Model Development:**

   * Train supervised models (including LSTM) for classification tasks.

5. **Evaluation and Optimization:**

   * Use multiple metrics to evaluate model performance and perform hyperparameter tuning.

6. **Deployment:**

   * Develop and deploy a real-time prediction prototype using Streamlit and AWS.

---

## Results Achieved

* **Tool Wear Detection:** High accuracy in identifying worn vs. unworn tools.
  
* **Machining Completion Prediction:** Reliable classification of process completion.
  
* **Inadequate Clamping Detection:** Early identification of insufficient clamping to avoid defects.

---

## Evaluation Metrics

* Accuracy

* AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
  
* Confusion Matrix Analysis
  
* Model Execution Time (critical for real-time deployment)

---

## Dataset Description

* **train.csv:** Summary data of each experiment.
  
* **experiment\_01.csv to experiment\_18.csv:** Detailed time series data from sensors during machining.

---

## Technologies and Tools Used

* Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)
  
* Long Short-Term Memory (LSTM) models
  
* Streamlit for UI development
  
* AWS for model hosting and deployment

---

## Conclusion

This project delivers a robust solution for predictive maintenance and fault detection in CNC milling operations by leveraging deep learning techniques, time series analysis, and interactive deployment.

---

> **Note:** This repository includes the code, models, and deployment files necessary to replicate or extend the work. Contributions and feedback are welcome!
