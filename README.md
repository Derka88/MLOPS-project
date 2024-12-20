# MLOps Project: Energy Consumption Prediction
Author : 
Ange metuengo Fotso
Nana Fatouma Abdou Dangaladima
Abdoul Kader Mamoudou kaka

## Project Overview
This project focuses on forecasting energy consumption using machine learning techniques, particularly time-series analysis. It integrates MLOps best practices for model development, deployment, and monitoring. The goal is to predict future energy usage based on historical data, environmental factors, and other relevant features.

## Project Structure

The project is organized as follows:
.
├── README.md
├── requirements.txt
├── Functions/
│   ├── import_data.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── xgboost_model.py
│   ├── Graphique.py
├── notebooks/
│   ├── Energy_Forecasting.ipynb
├── docs/
│   ├── Rapport_Projet_Mlops_Metuengo_Mamoudou_Abdou.pdf
└── data/


## Dependencies

To install the required libraries for this project, run:

```bash
pip install -r requirements.txt

```

Alternatively, if you're using Conda, you can create the environment from the environment.yml file (if available).

Required Libraries
numpy – for numerical operations
pandas – for data manipulation and analysis
scikit-learn – for machine learning models and evaluation
xgboost – for training and tuning XGBoost models
mlflow – for tracking experiments and model management
matplotlib – for data visualization
seaborn – for advanced visualizations
jupyter – for running the Jupyter notebook

Functionality
Functions
The Functions/ folder contains the following key modules:

import_data.py: Contains functions for importing and loading raw data into the project.
feature_engineering.py: Defines functions for transforming the raw data into meaningful features for machine learning models (e.g., extract_features).
models.py: Implements functions to train and evaluate various machine learning models.
xgboost_model.py: Provides functions to train and tune an XGBoost model, including hyperparameter optimization.
Graphique.py: Contains functions to visualize data, model performance, and predictions through graphs.
Notebooks
The notebooks/ directory includes:

Energy_Forecasting.ipynb: This Jupyter notebook is used for conducting exploratory data analysis (EDA), applying feature engineering, training machine learning models, and evaluating performance. It is the primary interface for running experiments and testing different model approaches.

How to Run the Project
Clone or download the project repository to your local machine.
Set up the Python environment by installing the necessary dependencies:
With pip: pip install -r requirements.txt
With conda: create a conda environment from the environment.yml file (if available).
Open the Energy_Forecasting.ipynb notebook in Jupyter.
Run the notebook cells in sequence to:
Load and preprocess the data.
Apply feature engineering techniques.
Train and evaluate machine learning models.
Visualize model predictions and results.
Modify the notebook as needed for different experiments, model tuning, or feature engineering approaches.

Methodology
For a detailed explanation of the methodology followed in this project, please refer to the docs/Rapport_Projet_Mlops_Metuengo_Mamoudou_Abdou file. This document outlines the steps taken from data collection to model deployment, including feature extraction, model selection, and evaluation strategies.

References
The docs/references.md file contains a list of all external resources and references used during the development of this project. These include tutorials, academic papers, and documentation for the libraries implemented.

Contributing
We welcome contributions to improve this project. If you wish to contribute, please fork the repository, create a new branch for your feature or fix, and submit a pull request. We encourage improvements in model performance, data handling, and visualization techniques.