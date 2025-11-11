#!/usr/bin/env python3
"""
Step 2: Train XGBoost Model
============================

This script demonstrates how to:
1. Load data from CSV or Snowflake
2. Split data into train/test sets
3. Train an XGBoost classifier with hyperparameter tuning
4. Evaluate model performance with multiple metrics
5. Display feature importance
6. Save the trained model for the next step

The trained model will be serialized in Step 3.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def load_data(source='csv', csv_path='synthetic_data.csv'):
    """
    Load data from CSV file or Snowflake table.
    
    Parameters:
    -----------
    source : str
        Data source: 'csv' or 'snowflake'
    csv_path : str
        Path to CSV file (if source='csv')
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    if source == 'csv':
        print(f"\nLoading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df):,} rows from CSV")
        
    elif source == 'snowflake':
        print(f"\nLoading data from Snowflake...")
        from connections import SnowflakeConnection
        
        connection = SnowflakeConnection.from_snow_cli('legalzoom')
        session = connection.session
        
        # Load from Snowflake table
        table_name = "ML_SHOWCASE.DATA.SYNTHETIC_DATA"
        df = session.table(table_name).to_pandas()
        print(f"✓ Loaded {len(df):,} rows from Snowflake")
        
        connection.close()
    else:
        raise ValueError(f"Invalid source: {source}. Use 'csv' or 'snowflake'")
    
    # Display data info
    print(f"\nData Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting into train/test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    print(f"\n{'=' * 80}")
    print("PREPARING DATA FOR TRAINING")
    print("=" * 80)
    
    # Separate features and target
    # Drop ID column if it exists
    feature_cols = [col for col in df.columns if col not in ['ID', 'TARGET']]
    X = df[feature_cols]
    y = df['TARGET']
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target: TARGET (binary: {y.nunique()} classes)")
    print(f"Class distribution:")
    for class_label, count in y.value_counts().sort_index().items():
        print(f"  Class {class_label}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\nData Split:")
    print(f"  Training set:   {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set:       {len(X_test):,} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_xgboost_model(X_train, y_train, tune_hyperparameters=True):
    """
    Train XGBoost classifier with optional hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    xgb.XGBClassifier
        Trained model
    """
    print(f"\n{'=' * 80}")
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        print("This may take a few minutes...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        
        print(f"\n✓ Hyperparameter tuning completed!")
        print(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
        
    else:
        print("\nTraining with default parameters...")
        
        # Train with reasonable default parameters
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print(f"✓ Model trained successfully!")
    
    # Display model parameters
    print(f"\nFinal Model Parameters:")
    print(f"  max_depth: {model.max_depth}")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  subsample: {model.subsample}")
    print(f"  colsample_bytree: {model.colsample_bytree}")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained model
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print("EVALUATING MODEL PERFORMANCE")
    print("=" * 80)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Predict probabilities for ROC-AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    }
    
    # Display metrics
    print("\nPerformance Metrics:")
    print(f"{'Metric':<15} {'Train':<12} {'Test':<12} {'Difference':<12}")
    print("-" * 55)
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        train_val = metrics['train'][metric_name]
        test_val = metrics['test'][metric_name]
        diff = train_val - test_val
        print(f"{metric_name.upper():<15} {train_val:<12.4f} {test_val:<12.4f} {diff:<12.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"Actual    0    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"          1    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Classification report
    print(f"\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1']))
    
    return metrics


def display_feature_importance(model, feature_names, top_n=10):
    """
    Display feature importance from the trained model.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """
    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} FEATURE IMPORTANCE")
    print("=" * 80)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print(f"\n{'Rank':<6} {'Feature':<20} {'Importance':<12} {'Bar'}")
    print("-" * 60)
    
    max_importance = importance_df['importance'].max()
    for idx, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        bar_length = int((row['importance'] / max_importance) * 30)
        bar = '█' * bar_length
        print(f"{idx:<6} {row['feature']:<20} {row['importance']:<12.4f} {bar}")
    
    return importance_df


def save_model_object(model, filename='trained_model.pkl'):
    """
    Save the trained model object to memory for next step.
    
    Note: This function saves the model in memory for use in the next script.
    In Step 3, we'll demonstrate proper serialization to a .pkl file.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained model
    filename : str
        Filename for reference (actual saving happens in Step 3)
        
    Returns:
    --------
    xgb.XGBClassifier
        The model object (to be used in next step)
    """
    print(f"\n{'=' * 80}")
    print("MODEL READY FOR SERIALIZATION")
    print("=" * 80)
    
    print(f"\nModel object is ready to be saved.")
    print(f"In the next step (03_save_pickle.py), we will:")
    print(f"  1. Serialize this model to a .pkl file")
    print(f"  2. Demonstrate both pickle and joblib methods")
    print(f"  3. Verify the model can be loaded back")
    
    return model


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("STEP 2: TRAIN XGBOOST MODEL")
    print("=" * 80)
    print("\nThis script trains an XGBoost classifier on the synthetic data.")
    
    # Configuration
    DATA_SOURCE = 'csv'  # 'csv' or 'snowflake'
    CSV_PATH = 'synthetic_data.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TUNE_HYPERPARAMETERS = False  # Set to True for hyperparameter tuning (slower)
    
    # Step 1: Load data
    df = load_data(source=DATA_SOURCE, csv_path=CSV_PATH)
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Step 3: Train model
    model = train_xgboost_model(
        X_train, 
        y_train, 
        tune_hyperparameters=TUNE_HYPERPARAMETERS
    )
    
    # Step 4: Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Display feature importance
    importance_df = display_feature_importance(model, feature_names, top_n=10)
    
    # Step 6: Prepare model for next step
    model_obj = save_model_object(model)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Model trained successfully")
    print(f"✓ Test Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"✓ Test F1 Score:  {metrics['test']['f1']:.4f}")
    print(f"✓ Test ROC-AUC:   {metrics['test']['roc_auc']:.4f}")
    
    # Save model and data for next steps
    print(f"\nSaving model and data for next steps...")
    
    # Save the trained model
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved trained model to: trained_model.pkl")
    
    # Save test data for inference examples
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('test_data.csv', index=False)
    print(f"✓ Saved test data to: test_data.csv")
    
    # Save metrics
    import json
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: model_metrics.json")
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print("Run the next script to serialize the model:")
    print("  python 03_save_pickle.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

