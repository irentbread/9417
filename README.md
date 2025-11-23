# Forecasting Air Pollution with Machine Learning
This project provides multiple methods of **regression** and **classification** on the **UCI Machine Learning Repository - Air Quality** Dataset

It includes **Ridge Regression**, **XGBoost** and **LSTM (Long Short-Term Memory)** models for regression.
It includes **LASSO Regression**, **XGBoost** and **FNN (Feed-Forward Neural Network)** models for classification

This project provides a pipeline for training, validating, and comparing different models on the Air Quality dataset.

## Installation
Create and activate a virtual environment, then install dependencies:
* `xgboost`
* `numpy`
* `scikit-learn`
* `torch`
* `polars`
* `pandas`
* `matplotlib`
* `shap`
* `tensorflow`
* `itertools`
* `seaborn`

> Python 3.8+ is recommended.

## Regression
### Machine Learning 
* **File(s)**:
  * `Machine Learning Ridge Regression Model.ipynb`: Jupyter Notebook implementing the Ridge model for regression of Air Quality.
  * `Machine Learning XGBoost Regression Model.ipynb`: Jupyter Notebook implementing the XGBoost model for regression of Air Quality.
* **Instructions to run**:
  * Run the cell in Jupyter Notebook for each ML regression model file, which will generate respective directories with relevant CSV file summary data and plots.
  * Preview of saved CSV files and Ridge test RMSE vs. Naive baseline RSME plots are shown.

### Deep Learning
* **File(s)**:
  * `Deep_Learning_LSTM_Regression_Model.py`: Main LSTM regression model file.
  * `Deep_Learning_LSTM_Regression_Baseline.py`
  * `Deep_Learning_LSTM_Regression_Plots.py`
  * `Deep_Learning_LSTM_Regression_Window_Generator.py`
* **Instructions to run**:
  * Run `Deep Learning LSTM Regression Model.py`, which will import the remaining supporting files and generate all necessary results.

## Classification
### Machine Learning 
* **File(s)**:
  * `Machine Learning LASSO XGBoost Classification Model.ipynb`: Jupyter Notebook implementing both the LASSO and XGBoost models for classification of Air Quality.
* **Instructions to run**:
  * Run all cells in Jupyter Notebook from end-to-end, including hyperparameter tuning and training and testing on validation and test sets.
  * All plots relevant to feature importance and performance against naive models are shown.
### Deep Learning
* **File(s)**:
  * `Deep Learning FNN Classification Model.ipynb`: Jupyter Notebook implementing the FNN model for classification of Air Quality.
  * `Sample limited temporal features FNN classification.ipynb`: Jupyter Notebook implementing the FNN model on limited temporal features for classification of Air Quality.

* **Instructions to run**:
  * Run all cells in Jupyter Notebook from end-to-end, including hyperparameter tuning and training and testing on validation and test sets.
  * All plots relevant to feature importance and performance against naive models are shown.
 
## Contact
For any questions, feel free to reach out.

| Name                 | Email                                                     |
| ---------------------|---------------------------------------------------------- |
| **Brad Pike**        | [z5679188@ad.unsw.edu.au](mailto:z5679188@ad.unsw.edu.au) |
| **Darren Xing**      | [z5478438@ad.unsw.edu.au](mailto:z5478438@ad.unsw.edu.au) |
| **Duncan Lai**       | [z5416904@ad.unsw.edu.au](mailto:z5416904@ad.unsw.edu.au) |
| **Joseph Zhu**       | [z5487476@ad.unsw.edu.au](mailto:z5487476@ad.unsw.edu.au) |
| **Muhanned Almadani**| [z5510596@ad.unsw.edu.au](mailto:z5510596@ad.unsw.edu.au) |
