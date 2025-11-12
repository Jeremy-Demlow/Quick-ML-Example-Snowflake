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
