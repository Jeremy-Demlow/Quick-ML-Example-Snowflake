#!/usr/bin/env python3
"""
Create sklearn Pipeline Model
==============================

This script creates an sklearn Pipeline that combines the scaler and model.
sklearn Pipelines ARE supported by Snowflake Model Registry.

We'll also save the individual pickles as user_files for reference/documentation.
"""

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from logging_utils import get_logger, log_section


logger = get_logger(__name__)


def create_pipeline_from_pickles():
    """
    Create an sklearn Pipeline from the separate pickle files.
    
    Returns:
    --------
    Pipeline
        sklearn Pipeline with scaler and model
    """
    log_section(logger, "CREATING SKLEARN PIPELINE FROM PICKLES")
    
    # Load the separate pickles
    logger.info("Loading scaler...")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded")
    
    logger.info("Loading model...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded")
    
    # Create a custom transformer that wraps our CustomZScaler
    class CustomZScalerTransformer(FunctionTransformer):
        def __init__(self, scaler):
            self.scaler = scaler
            super().__init__(func=self.transform_func)
        
        def transform_func(self, X):
            return self.scaler.transform(pd.DataFrame(X))
    
    # Create pipeline
    logger.info("Creating sklearn Pipeline...")
    pipeline = Pipeline([
        ('scaler', FunctionTransformer(
            func=lambda X: scaler.transform(pd.DataFrame(X, columns=scaler.feature_names_)),
            validate=False
        )),
        ('model', model)
    ])
    
    logger.info("Pipeline created")
    logger.info("  Steps: %s", [name for name, _ in pipeline.steps])
    
    # Save the pipeline
    logger.info("Saving pipeline...")
    with open('pipeline_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info("Pipeline saved to: pipeline_model.pkl")
    
    return pipeline


def test_pipeline(pipeline):
    """Test the pipeline works."""
    log_section(logger, "TESTING PIPELINE")
    
    # Load test data
    test_df = pd.read_csv('test_data.csv')
    feature_cols = [col for col in test_df.columns if col.startswith('FEATURE_')]
    X_test = test_df[feature_cols].head(5)
    y_test = test_df['TARGET'].head(5)
    
    logger.info("Making predictions on %s samples...", len(X_test))
    predictions = pipeline.predict(X_test)
    
    logger.info("Predictions successful!")
    logger.debug("Results:")
    for i, (pred, actual) in enumerate(zip(predictions, y_test)):
        logger.debug(
            "  Sample %s: Predicted=%.2f, Actual=%.2f, Error=%.2f",
            i,
            pred,
            actual,
            abs(pred - actual),
        )
    
    return True


if __name__ == "__main__":
    log_section(logger, "CREATE SKLEARN PIPELINE MODEL")
    logger.info("This creates an sklearn Pipeline from the separate pickles.")
    logger.info("sklearn Pipelines ARE supported by Snowflake Model Registry.")
    
    # Create pipeline
    pipeline = create_pipeline_from_pickles()
    
    # Test pipeline
    test_pipeline(pipeline)
    
    log_section(logger, "SUMMARY")
    logger.info("Pipeline created and saved: pipeline_model.pkl")
    logger.info("Pipeline tested successfully")
    logger.info("You can now log this pipeline to Model Registry:")
    logger.info("  - It's a standard sklearn Pipeline")
    logger.info("  - Include scaler.pkl and model.pkl as user_files for reference")

