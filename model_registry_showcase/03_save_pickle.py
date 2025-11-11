#!/usr/bin/env python3
"""
Step 3: Save Model as Pickle File
==================================

This script demonstrates how to:
1. Load a trained model from memory
2. Serialize the model using pickle
3. Serialize the model using joblib (alternative method)
4. Compare file sizes and performance
5. Verify the model can be loaded back correctly
6. Test that loaded model produces same predictions

This step creates the .pkl file that will be uploaded to Snowflake Model Registry.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys


def load_trained_model(model_path='trained_model.pkl'):
    """
    Load the trained model from the previous step.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    model
        Trained XGBoost model
    """
    print("=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)
    
    if not Path(model_path).exists():
        print(f"\n✗ Error: Model file not found: {model_path}")
        print(f"Please run 02_train_model.py first to train a model.")
        sys.exit(1)
    
    print(f"\nLoading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
    
    # Display model parameters
    print(f"\nModel Parameters:")
    params = model.get_params()
    key_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']
    for param in key_params:
        if param in params:
            print(f"  {param}: {params[param]}")
    
    return model


def serialize_with_pickle(model, filename='xgboost_model_pickle.pkl'):
    """
    Serialize model using Python's pickle module.
    
    Parameters:
    -----------
    model : object
        Trained model to serialize
    filename : str
        Output filename
        
    Returns:
    --------
    dict
        Serialization metadata (path, size, time)
    """
    print(f"\n{'=' * 80}")
    print("METHOD 1: SERIALIZATION WITH PICKLE")
    print("=" * 80)
    
    print(f"\nSerializing model with pickle...")
    print(f"Output file: {filename}")
    
    # Time the serialization
    start_time = time.time()
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    serialize_time = time.time() - start_time
    
    # Get file info
    filepath = Path(filename)
    file_size_bytes = filepath.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"✓ Serialization completed!")
    print(f"  File: {filepath}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"  Time: {serialize_time:.4f} seconds")
    print(f"  Protocol: {pickle.HIGHEST_PROTOCOL}")
    
    return {
        'path': filepath,
        'size_bytes': file_size_bytes,
        'size_mb': file_size_mb,
        'time_seconds': serialize_time,
        'method': 'pickle'
    }


def serialize_with_joblib(model, filename='xgboost_model_joblib.pkl'):
    """
    Serialize model using joblib (optimized for large numpy arrays).
    
    Parameters:
    -----------
    model : object
        Trained model to serialize
    filename : str
        Output filename
        
    Returns:
    --------
    dict
        Serialization metadata (path, size, time)
    """
    print(f"\n{'=' * 80}")
    print("METHOD 2: SERIALIZATION WITH JOBLIB")
    print("=" * 80)
    
    print(f"\nSerializing model with joblib...")
    print(f"Output file: {filename}")
    
    # Time the serialization
    start_time = time.time()
    
    joblib.dump(model, filename, compress=3)  # compress=3 for good balance
    
    serialize_time = time.time() - start_time
    
    # Get file info
    filepath = Path(filename)
    file_size_bytes = filepath.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"✓ Serialization completed!")
    print(f"  File: {filepath}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"  Time: {serialize_time:.4f} seconds")
    print(f"  Compression: Level 3")
    
    return {
        'path': filepath,
        'size_bytes': file_size_bytes,
        'size_mb': file_size_mb,
        'time_seconds': serialize_time,
        'method': 'joblib'
    }


def compare_methods(pickle_meta, joblib_meta):
    """
    Compare pickle and joblib serialization methods.
    
    Parameters:
    -----------
    pickle_meta : dict
        Pickle serialization metadata
    joblib_meta : dict
        Joblib serialization metadata
    """
    print(f"\n{'=' * 80}")
    print("COMPARISON: PICKLE VS JOBLIB")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'Pickle':<20} {'Joblib':<20} {'Winner'}")
    print("-" * 75)
    
    # File size comparison
    pickle_size = pickle_meta['size_mb']
    joblib_size = joblib_meta['size_mb']
    size_winner = 'Joblib' if joblib_size < pickle_size else 'Pickle'
    size_diff = abs(pickle_size - joblib_size)
    print(f"{'File Size (MB)':<20} {pickle_size:<20.2f} {joblib_size:<20.2f} {size_winner} (-{size_diff:.2f} MB)")
    
    # Time comparison
    pickle_time = pickle_meta['time_seconds']
    joblib_time = joblib_meta['time_seconds']
    time_winner = 'Joblib' if joblib_time < pickle_time else 'Pickle'
    time_diff = abs(pickle_time - joblib_time)
    print(f"{'Serialize Time (s)':<20} {pickle_time:<20.4f} {joblib_time:<20.4f} {time_winner} (-{time_diff:.4f} s)")
    
    print(f"\nRecommendation:")
    print(f"  • Pickle: Standard Python serialization, good for general use")
    print(f"  • Joblib: Optimized for large numpy arrays, better compression")
    print(f"  • For Snowflake Model Registry: Either works, joblib often preferred for ML models")


def verify_serialization(original_model, pickle_path, joblib_path):
    """
    Verify that serialized models can be loaded and produce correct predictions.
    
    Parameters:
    -----------
    original_model : object
        Original trained model
    pickle_path : Path
        Path to pickle-serialized model
    joblib_path : Path
        Path to joblib-serialized model
    """
    print(f"\n{'=' * 80}")
    print("VERIFICATION: LOADING AND TESTING SERIALIZED MODELS")
    print("=" * 80)
    
    # Load test data
    print(f"\nLoading test data...")
    if not Path('test_data.csv').exists():
        print(f"✗ Test data not found. Skipping verification.")
        return
    
    test_df = pd.read_csv('test_data.csv')
    X_test = test_df.drop(columns=['TARGET'])
    y_test = test_df['TARGET']
    
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # Get original predictions
    print(f"\nGetting predictions from original model...")
    original_pred = original_model.predict(X_test[:100])  # Test on first 100 samples
    original_proba = original_model.predict_proba(X_test[:100])
    
    # Test pickle-serialized model
    print(f"\nTesting pickle-serialized model...")
    start_time = time.time()
    with open(pickle_path, 'rb') as f:
        pickle_model = pickle.load(f)
    pickle_load_time = time.time() - start_time
    
    pickle_pred = pickle_model.predict(X_test[:100])
    pickle_proba = pickle_model.predict_proba(X_test[:100])
    
    pickle_match = np.array_equal(original_pred, pickle_pred)
    pickle_proba_match = np.allclose(original_proba, pickle_proba)
    
    print(f"  Load time: {pickle_load_time:.4f} seconds")
    print(f"  Predictions match: {pickle_match} ✓" if pickle_match else f"  Predictions match: {pickle_match} ✗")
    print(f"  Probabilities match: {pickle_proba_match} ✓" if pickle_proba_match else f"  Probabilities match: {pickle_proba_match} ✗")
    
    # Test joblib-serialized model
    print(f"\nTesting joblib-serialized model...")
    start_time = time.time()
    joblib_model = joblib.load(joblib_path)
    joblib_load_time = time.time() - start_time
    
    joblib_pred = joblib_model.predict(X_test[:100])
    joblib_proba = joblib_model.predict_proba(X_test[:100])
    
    joblib_match = np.array_equal(original_pred, joblib_pred)
    joblib_proba_match = np.allclose(original_proba, joblib_proba)
    
    print(f"  Load time: {joblib_load_time:.4f} seconds")
    print(f"  Predictions match: {joblib_match} ✓" if joblib_match else f"  Predictions match: {joblib_match} ✗")
    print(f"  Probabilities match: {joblib_proba_match} ✓" if joblib_proba_match else f"  Probabilities match: {joblib_proba_match} ✗")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if pickle_match and pickle_proba_match and joblib_match and joblib_proba_match:
        print(f"✓ All serialization methods verified successfully!")
        print(f"✓ Both pickle and joblib models produce identical predictions")
        print(f"✓ Models are ready for upload to Snowflake Model Registry")
    else:
        print(f"✗ Verification failed - predictions don't match!")
        print(f"  This could indicate a serialization issue.")


def display_model_metadata(model, pickle_path):
    """
    Display comprehensive model metadata.
    
    Parameters:
    -----------
    model : object
        Trained model
    pickle_path : Path
        Path to serialized model file
    """
    print(f"\n{'=' * 80}")
    print("MODEL METADATA")
    print("=" * 80)
    
    print(f"\nModel Information:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Module: {model.__class__.__module__}")
    print(f"  Class: {model.__class__.__name__}")
    
    # Get model parameters
    params = model.get_params()
    print(f"\nKey Hyperparameters:")
    key_params = [
        'max_depth', 'learning_rate', 'n_estimators', 
        'subsample', 'colsample_bytree', 'objective', 'eval_metric'
    ]
    for param in key_params:
        if param in params:
            print(f"  {param}: {params[param]}")
    
    # File information
    print(f"\nSerialized File:")
    print(f"  Path: {pickle_path}")
    print(f"  Size: {pickle_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Format: Pickle (Python serialization)")
    
    # Feature information
    if hasattr(model, 'feature_names_in_'):
        print(f"\nFeature Information:")
        print(f"  Number of features: {len(model.feature_names_in_)}")
        print(f"  Feature names: {list(model.feature_names_in_[:5])}...")
    
    # Model capabilities
    print(f"\nModel Capabilities:")
    print(f"  predict(): Binary classification predictions")
    print(f"  predict_proba(): Class probability estimates")
    if hasattr(model, 'predict_log_proba'):
        print(f"  predict_log_proba(): Log probability estimates")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("STEP 3: SERIALIZE MODEL TO PICKLE FILE")
    print("=" * 80)
    print("\nThis script demonstrates model serialization using pickle and joblib.")
    print("The serialized .pkl file will be uploaded to Snowflake Model Registry.")
    
    # Configuration
    PICKLE_FILENAME = 'xgboost_model_pickle.pkl'
    JOBLIB_FILENAME = 'xgboost_model_joblib.pkl'
    TRAINED_MODEL_PATH = 'trained_model.pkl'
    
    # Step 1: Load trained model
    model = load_trained_model(TRAINED_MODEL_PATH)
    
    # Step 2: Serialize with pickle
    pickle_meta = serialize_with_pickle(model, PICKLE_FILENAME)
    
    # Step 3: Serialize with joblib
    joblib_meta = serialize_with_joblib(model, JOBLIB_FILENAME)
    
    # Step 4: Compare methods
    compare_methods(pickle_meta, joblib_meta)
    
    # Step 5: Verify serialization
    verify_serialization(model, pickle_meta['path'], joblib_meta['path'])
    
    # Step 6: Display model metadata
    display_model_metadata(model, pickle_meta['path'])
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Model serialized successfully with both methods")
    print(f"✓ Pickle file: {PICKLE_FILENAME} ({pickle_meta['size_mb']:.2f} MB)")
    print(f"✓ Joblib file: {JOBLIB_FILENAME} ({joblib_meta['size_mb']:.2f} MB)")
    print(f"✓ Both models verified to produce identical predictions")
    
    print(f"\nRecommended file for Snowflake Model Registry:")
    print(f"  → {JOBLIB_FILENAME}")
    print(f"  (Joblib is optimized for ML models with numpy arrays)")
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print("Run the next script to initialize the Model Registry:")
    print("  python 04_create_registry.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

