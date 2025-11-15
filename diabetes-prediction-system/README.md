# Diabetes Prediction Using Ensemble Machine Learning Pipeline

This project builds a complete end-to-end **Diabetes Prediction Model**
using an ensemble of machine learning algorithms packed inside a
**Voting Classifier Pipeline**.\
It covers data preprocessing, exploratory data analysis (EDA), model
training, evaluation, feature importance, model persistence, and
testing.

## 🚀 Project Overview

This project predicts whether a patient has diabetes based on diagnostic
measurements.\
It uses:

-   Data Cleaning
-   Missing Value Handling (Imputation)
-   Normalization (Min-Max Scaling)
-   Ensemble Learning (Voting Classifier)
-   Model Evaluation & Visualization
-   Saving & Loading Trained Models

The dataset used is **diabetes.csv** (Pima Indians Diabetes Dataset).

## 🧠 Machine Learning Models Used

The Voting Classifier combines:

1.  Decision Tree Classifier\
2.  K-Nearest Neighbors (KNN)\
3.  Logistic Regression

Voting Strategy → **Soft Voting**

## 📁 Project Structure

    ├── diabetes.csv
    ├── diabetes_model_pipeline.joblib
    ├── main.py
    └── README.md

## 🛠️ Key Features

### ✔ Data Preprocessing

-   Replaces zero values (medically impossible) with NaN\
-   Imputes missing values using SimpleImputer\
-   Normalizes all feature values using MinMaxScaler

### ✔ Exploratory Data Analysis (EDA)

-   Outcome distribution plot\
-   Pairplot for feature relationships

### ✔ Model Training Pipeline

-   Preprocessing + Voting Classifier combined in sklearn Pipeline

### ✔ Model Evaluation

-   Classification report\
-   Confusion matrix\
-   Feature importance\
-   Individual model evaluation

### ✔ Save & Load Model

-   Saves the entire ML pipeline using joblib\
-   Supports predictions after loading

## 📊 Dataset Information

Features: - Pregnancies\
- Glucose\
- BloodPressure\
- SkinThickness\
- Insulin\
- BMI\
- DiabetesPedigreeFunction\
- Age

Target: - Outcome (1 = Diabetes, 0 = No Diabetes)

## ▶️ How to Run

Install packages:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Run:

``` bash
python main.py
```

## 👨‍💻 Author

**Shubham Londhe**\
Date: 10/11/2025
