#!/usr/bin/env python3
"""
Custom Z-Scaler and Model Training
===================================

This module demonstrates how to create:
1. Custom Z-scaler (StandardScaler) preprocessing
2. Linear regression model
3. Separate pickle files for each component
4. Custom predict function that uses both

This approach uses separate pickle files that will be included via
the `user_files` parameter in registry.log_model().
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Dict, Any
import pickle
import os

from logging_utils import get_logger, log_section


logger = get_logger(__name__)


class CustomZScaler:
    """
    Custom Z-Score Scaler (StandardScaler implementation).
    
    This demonstrates how to create a custom preprocessor that can be
    serialized and used within a Snowflake Model Registry model.
    
    The Z-scaler transforms features by:
    - Removing the mean (centering)
    - Scaling to unit variance
    
    Formula: z = (x - mean) / std
    
    Attributes:
    -----------
    mean_ : np.ndarray
        Mean of each feature from training data
    std_ : np.ndarray
        Standard deviation of each feature from training data
    feature_names_ : list
        Names of features (column names)
    """
    
    def __init__(self):
        """Initialize the Z-scaler."""
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame) -> 'CustomZScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data to compute statistics
            
        Returns:
        --------
        self : CustomZScaler
            Fitted scaler
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            self.feature_names_ = None
        
        self.mean_ = np.mean(X_values, axis=0)
        self.std_ = np.std(X_values, axis=0)
        
        # Handle zero std (constant features)
        self.std_[self.std_ == 0] = 1.0
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform
            
        Returns:
        --------
        X_scaled : pd.DataFrame
            Transformed data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            columns = X.columns
            index = X.index
        else:
            X_values = X
            columns = self.feature_names_
            index = None
        
        # Apply z-score transformation
        X_scaled = (X_values - self.mean_) / self.std_
        
        # Return as DataFrame with original column names
        if columns is not None:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        else:
            return pd.DataFrame(X_scaled)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to fit and transform
            
        Returns:
        --------
        X_scaled : pd.DataFrame
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters of the scaler.
        
        Returns:
        --------
        params : dict
            Dictionary of parameters
        """
        return {
            'mean': self.mean_.tolist() if self.mean_ is not None else None,
            'std': self.std_.tolist() if self.std_ is not None else None,
            'feature_names': self.feature_names_,
            'is_fitted': self.is_fitted_
        }


def train_model_with_preprocessing(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train model with preprocessing and save separate pickle files.
    
    This function:
    1. Creates and fits the custom Z-scaler
    2. Transforms training data
    3. Trains linear regression model
    4. Saves scaler and model as separate pickle files
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.ndarray
        Training targets
        
    Returns:
    --------
    scaler : CustomZScaler
        Fitted scaler
    model : LinearRegression
        Fitted model
    """
    log_section(logger, "TRAINING MODEL WITH PREPROCESSING")
    
    # Step 1: Create and fit the custom Z-scaler
    logger.info("1. Fitting Custom Z-Scaler...")
    scaler = CustomZScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info("   Scaler fitted")
    logger.info("   Features: %s", len(scaler.feature_names_))
    logger.info(
        "   Mean range: [%.2f, %.2f]",
        scaler.mean_.min(),
        scaler.mean_.max(),
    )
    logger.info(
        "   Std range: [%.2f, %.2f]",
        scaler.std_.min(),
        scaler.std_.max(),
    )
    
    # Step 2: Train linear regression model
    logger.info("2. Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    logger.info("   Model trained")
    logger.info("   Coefficients: %s", len(model.coef_))
    logger.info("   Intercept: %.2f", model.intercept_)
    
    # Step 3: Save as separate pickle files
    logger.info("3. Saving Components as Separate Pickle Files...")
    
    scaler_path = "scaler.pkl"
    model_path = "model.pkl"
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("   Saved scaler to: %s", scaler_path)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("   Saved model to: %s", model_path)
    
    logger.info("Training complete!")
    logger.info("  Components saved as separate pickle files:")
    logger.info("    - %s (preprocessing)", scaler_path)
    logger.info("    - %s (model)", model_path)
    
    return scaler, model


# Custom predict function that will be used in Snowflake
def custom_predict(X: pd.DataFrame, scaler_path: str = 'scaler.pkl', model_path: str = 'model.pkl') -> pd.DataFrame:
    """
    Custom predict function that loads pickles and makes predictions.
    
    This function will be used by the Model Registry for inference.
    It loads the scaler and model from pickle files, applies preprocessing,
    and returns predictions.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    scaler_path : str
        Path to scaler pickle file
    model_path : str
        Path to model pickle file
        
    Returns:
    --------
    predictions : pd.DataFrame
        DataFrame with 'PREDICTION' column
    """
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Apply preprocessing
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Return as DataFrame
    return pd.DataFrame({'PREDICTION': predictions})


if __name__ == "__main__":
    """
    Test the custom model locally.
    """
    log_section(logger, "TESTING CUSTOM MODEL LOCALLY")
    
    # Generate sample data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"FEATURE_{i}" for i in range(5)])
    
    logger.info("Generated test data:")
    logger.info("  Samples: %s", len(X_df))
    logger.info("  Features: %s", X_df.shape[1])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Train and save model
    scaler, model = train_model_with_preprocessing(X_train, y_train)
    
    # Test custom predict function
    log_section(logger, "TESTING CUSTOM PREDICT FUNCTION")
    predictions = custom_predict(X_test)
    
    logger.info("Predictions shape: %s", predictions.shape)
    logger.debug("First 5 predictions:\n%s", predictions.head())
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, predictions['PREDICTION'])
    r2 = r2_score(y_test, predictions['PREDICTION'])
    
    logger.info("Model Performance:")
    logger.info("  MSE: %.2f", mse)
    logger.info("  R²:  %.4f", r2)
    
    logger.info("Custom model test completed successfully!")
