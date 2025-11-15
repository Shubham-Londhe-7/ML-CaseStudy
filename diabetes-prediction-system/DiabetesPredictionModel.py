#############################################################################################################
# Required Python Packages
#############################################################################################################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#############################################################################################################
# File Paths
#############################################################################################################
INPUT_PATH = 'diabetes.csv'
MODEL_PATH = "diabetes_model_pipeline.joblib"


#############################################################################################################
# Headers
#############################################################################################################
HEADERS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


#############################################################################################################
# Function name :       dataset_statistics
# Description :         Display comprehensive dataset statistics
# Input :               Dataset with related information
# Output :              Statistical summary printed to console
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def dataset_statistics(dataset):
    """Display basic statistical information about the dataset"""
    border = '-' * 120
    
    print(border)
    print("DATASET STATISTICS")
    print(border)
    print(f"Dimension : {dataset.shape}")
    print(f"\nFirst 5 entries :")
    print(dataset.head())
    
    print(f"\nChecking for null values in the Dataset :")
    print(dataset.isna().sum())
    
    print(f"\nStatistical Summary :")
    print(dataset.describe())
    print(border)


#############################################################################################################
# Function name :       handle_zero_values
# Description :         Replace zero values with NaN for features where zero is biologically impossible
# Input :               Dataset with potential zero values
# Output :              Dataset with zeros replaced by NaN
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def handle_zero_values(dataset):
    """
    Replace zero values with NaN for medical features where zero is not realistic.
    Features: Glucose, BloodPressure, SkinThickness, Insulin, BMI
    """
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_columns:
        dataset[col] = dataset[col].replace(0, np.nan)
    
    print(f"\nZero values replaced with NaN for features: {zero_columns}")
    print(f"Missing values after replacement:")
    print(dataset[zero_columns].isna().sum())
    
    return dataset


#############################################################################################################
# Function name :       visualize_data
# Description :         Create visualizations for exploratory data analysis
# Input :               Dataset and target variable
# Output :              Display plots for data distribution and relationships
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def visualize_data(dataset, target_col='Outcome'):
    """Generate EDA visualizations"""
    
    # Distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(data=dataset, x=target_col, bins=3, kde=False)
    plt.xlabel("Distribution of the Dependent Variable (Outcome)")
    plt.ylabel("Frequency of Distribution")
    plt.title("Diabetes Outcome Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Pairplot for feature relationships
    print("\nGenerating pairplot... (this may take a moment)")
    sns.pairplot(data=dataset, hue=target_col, diag_kind='kde', corner=True)
    plt.suptitle("Feature Relationships by Diabetes Outcome", y=1.01)
    plt.tight_layout()
    plt.show()


#############################################################################################################
# Function name :       split_dataset
# Description :         Split the dataset into training and testing sets
# Input :               Features (X), Target (Y), and test set percentage
# Output :              X_train, X_test, Y_train, Y_test
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def split_dataset(X, Y, test_percentage=0.2, random_state=42):
    """Split dataset into train and test sets"""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_percentage, random_state=random_state, stratify=Y
    )
    
    print(f"\nDataset Split:")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"Y_train shape : {Y_train.shape}")
    print(f"Y_test shape  : {Y_test.shape}")
    
    return X_train, X_test, Y_train, Y_test


#############################################################################################################
# Function name :       build_pipeline
# Description :         Build a machine learning pipeline with VotingClassifier
# Pipeline Steps :      
#                       1. SimpleImputer - Handle missing values with mean strategy
#                       2. MinMaxScaler - Normalize features to [0, 1] range
#                       3. VotingClassifier - Ensemble of DecisionTree, KNN, LogisticRegression
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def build_pipeline():
    """
    Build a pipeline with preprocessing and VotingClassifier.
    Uses soft voting to average predicted probabilities.
    """
    
    # Define base estimators
    dt_clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    knn_clf = KNeighborsClassifier(
        n_neighbors=13,
        weights='distance',
        metric='euclidean'
    )
    
    lr_clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    # Create VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[
            ('dt', dt_clf),
            ('knn', knn_clf),
            ('lr', lr_clf)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Build pipeline
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('voting', voting_clf)
    ])
    
    return pipeline


#############################################################################################################
# Function name :       train_pipeline
# Description :         Train the machine learning pipeline
# Input :               Pipeline object, training features and labels
# Output :              Trained pipeline
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def train_pipeline(pipeline, X_train, Y_train):
    """Fit the pipeline on training data"""
    print("\nTraining the model pipeline...")
    pipeline.fit(X_train, Y_train)
    print("Training completed successfully!")
    return pipeline


#############################################################################################################
# Function name :       evaluate_model
# Description :         Evaluate model performance on train and test sets
# Input :               Trained model, test/train data
# Output :              Print accuracy, classification report, and confusion matrix
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    """Comprehensive model evaluation"""
    border = '-' * 120
    
    # Predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    
    print(f"\n{border}")
    print("MODEL EVALUATION - VOTING CLASSIFIER")
    print(f"{border}")
    print(f"Training Accuracy   : {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy    : {test_accuracy * 100:.2f}%")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(Y_test, Y_test_pred, target_names=['No Diabetes', 'Diabetes']))
    
    print(f"\nConfusion Matrix (Test Set):")
    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    print(conf_matrix)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Diabetes Prediction')
    plt.tight_layout()
    plt.show()
    
    print(f"{border}")


#############################################################################################################
# Function name :       evaluate_individual_models
# Description :         Evaluate each model in the voting classifier separately
# Input :               Trained pipeline, test data
# Output :              Individual model accuracies
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def evaluate_individual_models(pipeline, X_train, X_test, Y_train, Y_test):
    """Evaluate individual models within the VotingClassifier"""
    border = '-' * 120
    
    print(f"\n{border}")
    print("INDIVIDUAL MODEL PERFORMANCE")
    print(f"{border}")
    
    # Get the voting classifier
    voting_clf = pipeline.named_steps['voting']
    
    # Transform data through preprocessing steps
    X_train_transformed = pipeline.named_steps['imputer'].transform(X_train)
    X_train_transformed = pipeline.named_steps['scaler'].transform(X_train_transformed)
    
    X_test_transformed = pipeline.named_steps['imputer'].transform(X_test)
    X_test_transformed = pipeline.named_steps['scaler'].transform(X_test_transformed)
    
    # Evaluate each estimator
    for name, estimator in voting_clf.named_estimators_.items():
        train_pred = estimator.predict(X_train_transformed)
        test_pred = estimator.predict(X_test_transformed)
        
        train_acc = accuracy_score(Y_train, train_pred)
        test_acc = accuracy_score(Y_test, test_pred)
        
        model_name = {
            'dt': 'Decision Tree',
            'knn': 'K-Nearest Neighbors',
            'lr': 'Logistic Regression'
        }[name]
        
        print(f"\n{model_name}:")
        print(f"  Training Accuracy : {train_acc * 100:.2f}%")
        print(f"  Testing Accuracy  : {test_acc * 100:.2f}%")
    
    print(f"{border}")


#############################################################################################################
# Function name :       plot_feature_importances
# Description :         Display feature importance from Decision Tree in VotingClassifier
# Input :               Trained pipeline, feature names
# Output :              Feature importance bar plot
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def plot_feature_importances(pipeline, feature_names):
    """Plot feature importances from the Decision Tree model"""
    
    # Extract Decision Tree from VotingClassifier
    voting_clf = pipeline.named_steps['voting']
    dt_model = voting_clf.named_estimators_['dt']
    
    if hasattr(dt_model, 'feature_importances_'):
        importances = dt_model.feature_importances_
        idx = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[idx], color='steelblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in idx], 
                   rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.xlabel("Features")
        plt.title("Feature Importance - Decision Tree (from Voting Classifier)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importances not available")


#############################################################################################################
# Function name :       save_model
# Description :         Save the trained model pipeline to disk
# Input :               Trained model, file path
# Output :              Saved model file
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def save_model(model, path=MODEL_PATH):
    """Save model pipeline using joblib"""
    joblib.dump(model, path)
    print(f"\nModel successfully saved to: {path}")


#############################################################################################################
# Function name :       load_model
# Description :         Load a trained model pipeline from disk
# Input :               File path
# Output :              Loaded model pipeline
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def load_model(path=MODEL_PATH):
    """Load model pipeline using joblib"""
    model = joblib.load(path)
    print(f"\nModel successfully loaded from: {path}")
    return model


#############################################################################################################
# Function name :       test_loaded_model
# Description :         Test the loaded model with sample predictions
# Input :               Loaded model, test data
# Output :              Sample prediction results
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def test_loaded_model(model, X_test, Y_test):
    """Test loaded model with sample predictions"""
    border = '-' * 120
    
    print(f"\n{border}")
    print("TESTING LOADED MODEL")
    print(f"{border}")
    
    # Test with first 5 samples
    sample_data = X_test.iloc[:5]
    sample_labels = Y_test.iloc[:5]
    
    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)
    
    print("\nSample Predictions:")
    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, sample_labels)):
        print(f"\nSample {i+1}:")
        print(f"  Predicted: {'Diabetes' if pred == 1 else 'No Diabetes'}")
        print(f"  Actual: {'Diabetes' if actual == 1 else 'No Diabetes'}")
        print(f"  Probability: [No Diabetes: {prob[0]:.3f}, Diabetes: {prob[1]:.3f}]")
        print(f"  Match: {'✓' if pred == actual else '✗'}")
    
    print(f"{border}")


#############################################################################################################
# Function name :       main
# Description :         Main function from where execution starts
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def main():
    """Main execution function"""
    border = '-' * 120
    
    print(f"\n{border}")
    print(" " * 40 + "DIABETES PREDICTION MODEL")
    print(f"{border}\n")
    
    # 1) Load dataset
    print("Loading dataset...")
    dataset = pd.read_csv(INPUT_PATH)
    
    # 2) Display statistics
    dataset_statistics(dataset)
    
    # 3) Handle zero values (replace with NaN for imputation)
    dataset = handle_zero_values(dataset)
    
    # 4) Visualizations
    visualize_data(dataset)
    
    # 5) Prepare features and target
    feature_headers = HEADERS[:-1]  # All except 'Outcome'
    target_header = HEADERS[-1]      # 'Outcome'
    
    X = dataset[feature_headers]
    Y = dataset[target_header]
    
    # 6) Split dataset
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, test_percentage=0.2)
    
    # 7) Build and train pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, X_train, Y_train)
    
    # 8) Evaluate ensemble model
    evaluate_model(trained_model, X_train, Y_train, X_test, Y_test)
    
    # 9) Evaluate individual models
    evaluate_individual_models(trained_model, X_train, X_test, Y_train, Y_test)
    
    # 10) Plot feature importances
    plot_feature_importances(trained_model, feature_headers)
    
    # 11) Save model
    save_model(trained_model, MODEL_PATH)
    
    # 12) Load and test saved model
    loaded_model = load_model(MODEL_PATH)
    test_loaded_model(loaded_model, X_test, Y_test)
    
    print(f"\n{border}")
    print(" " * 45 + "PROGRAM TERMINATED")
    print(f"{border}\n")


#############################################################################################################
# Application Starter
#############################################################################################################
if __name__ == "__main__":
    main()