#!/usr/bin/env python3
"""
Step 1: Generate Synthetic Data
================================

This script demonstrates how to:
1. Create a synthetic classification dataset using scikit-learn
2. Save the data locally as CSV
3. Upload the data to a Snowflake table
4. Display data statistics and schema

The generated data will be used for training an XGBoost model in the next step.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path
import sys

# Import Snowflake connection helper
from connections import SnowflakeConnection


def generate_synthetic_data(n_samples=10000, n_features=20, random_state=42):
    """
    Generate synthetic classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Total number of features
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and target column
    """
    print("=" * 80)
    print("GENERATING SYNTHETIC CLASSIFICATION DATA")
    print("=" * 80)
    
    # Generate classification dataset
    # - n_informative: number of informative features
    # - n_redundant: number of redundant features (linear combinations)
    # - n_repeated: number of duplicated features
    # - n_clusters_per_class: number of clusters per class
    # - class_sep: larger values spread out the classes
    # - flip_y: fraction of samples with random labels (noise)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,          # 15 informative features
        n_redundant=3,             # 3 redundant features
        n_repeated=0,              # No repeated features
        n_classes=2,               # Binary classification
        n_clusters_per_class=2,    # 2 clusters per class
        class_sep=1.0,             # Moderate class separation
        flip_y=0.01,               # 1% label noise
        random_state=random_state
    )
    
    # Create DataFrame with meaningful column names
    feature_names = [f"FEATURE_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['TARGET'] = y
    
    # Add an ID column
    df.insert(0, 'ID', range(1, len(df) + 1))
    
    print(f"\nDataset created successfully!")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features}")
    print(f"  Classes: 2 (binary classification)")
    
    # Display class distribution
    class_counts = df['TARGET'].value_counts().sort_index()
    print(f"\nClass Distribution:")
    for class_label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {class_label}: {count:,} samples ({percentage:.1f}%)")
    
    # Display basic statistics
    print(f"\nFeature Statistics:")
    print(f"  Mean: {df[feature_names].mean().mean():.4f}")
    print(f"  Std:  {df[feature_names].std().mean():.4f}")
    print(f"  Min:  {df[feature_names].min().min():.4f}")
    print(f"  Max:  {df[feature_names].max().max():.4f}")
    
    return df


def save_to_csv(df, filename='synthetic_data.csv'):
    """
    Save DataFrame to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Output filename
        
    Returns:
    --------
    Path
        Path to the saved file
    """
    print(f"\n{'=' * 80}")
    print("SAVING DATA TO CSV")
    print("=" * 80)
    
    filepath = Path(filename)
    df.to_csv(filepath, index=False)
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\nData saved to: {filepath}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    # Display first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return filepath


def upload_to_snowflake(df, connection_name='legalzoom', 
                        database='ML_SHOWCASE', 
                        schema='DATA',
                        table_name='SYNTHETIC_DATA'):
    """
    Upload DataFrame to Snowflake table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to upload
    connection_name : str
        Snowflake connection name
    database : str
        Target database name
    schema : str
        Target schema name
    table_name : str
        Target table name
        
    Returns:
    --------
    str
        Fully qualified table name
    """
    print(f"\n{'=' * 80}")
    print("UPLOADING DATA TO SNOWFLAKE")
    print("=" * 80)
    
    try:
        # Connect to Snowflake
        print(f"\nConnecting to Snowflake...")
        connection = SnowflakeConnection.from_snow_cli(connection_name)
        session = connection.session
        
        # Create database and schema if they don't exist
        print(f"Setting up database and schema...")
        session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()
        session.sql(f"USE DATABASE {database}").collect()
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}").collect()
        session.sql(f"USE SCHEMA {schema}").collect()
        
        # Convert pandas DataFrame to Snowpark DataFrame and save as table
        print(f"Uploading data to {database}.{schema}.{table_name}...")
        snowpark_df = session.create_dataframe(df)
        
        # Write to table (overwrite if exists)
        snowpark_df.write.mode("overwrite").save_as_table(table_name)
        
        # Verify the upload
        row_count = session.table(table_name).count()
        
        print(f"\n✓ Data uploaded successfully!")
        print(f"  Table: {database}.{schema}.{table_name}")
        print(f"  Rows: {row_count:,}")
        
        # Display table schema
        print(f"\nTable Schema:")
        schema_info = session.sql(f"DESCRIBE TABLE {table_name}").collect()
        for row in schema_info[:5]:  # Show first 5 columns
            print(f"  {row['name']}: {row['type']}")
        if len(schema_info) > 5:
            print(f"  ... and {len(schema_info) - 5} more columns")
        
        # Display sample data from Snowflake
        print(f"\nSample data from Snowflake:")
        sample_df = session.table(table_name).limit(3).to_pandas()
        print(sample_df)
        
        # Close connection
        connection.close()
        
        return f"{database}.{schema}.{table_name}"
        
    except Exception as e:
        print(f"\n✗ Error uploading to Snowflake: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE SYNTHETIC DATA")
    print("=" * 80)
    print("\nThis script creates synthetic data for ML model training.")
    print("The data will be saved locally and uploaded to Snowflake.")
    
    # Configuration
    N_SAMPLES = 10000
    N_FEATURES = 20
    RANDOM_STATE = 42
    CSV_FILENAME = 'synthetic_data.csv'
    
    # Step 1: Generate data
    df = generate_synthetic_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        random_state=RANDOM_STATE
    )
    
    # Step 2: Save to CSV
    csv_path = save_to_csv(df, filename=CSV_FILENAME)
    
    # Step 3: Upload to Snowflake
    table_name = upload_to_snowflake(df)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Generated {len(df):,} samples with {N_FEATURES} features")
    print(f"✓ Saved to CSV: {csv_path}")
    if table_name:
        print(f"✓ Uploaded to Snowflake: {table_name}")
    else:
        print(f"✗ Snowflake upload failed (check connection)")
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print("Run the next script to train an XGBoost model:")
    print("  python 02_train_model.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

