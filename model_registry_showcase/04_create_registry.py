#!/usr/bin/env python3
"""
Step 4: Initialize Snowflake Model Registry
============================================

This script demonstrates how to:
1. Connect to Snowflake using Snowpark
2. Create database and schema for Model Registry
3. Initialize the Snowflake Model Registry
4. Understand Registry parameters and configuration
5. Check privileges and permissions
6. List existing models (if any)

The Model Registry is a schema-level object that manages ML models in Snowflake.
"""

from snowflake.ml.registry import Registry
from connections import SnowflakeConnection
import sys


def connect_to_snowflake(connection_name='legalzoom'):
    """
    Establish connection to Snowflake.
    
    Parameters:
    -----------
    connection_name : str
        Connection name from Snow CLI configuration
        
    Returns:
    --------
    tuple
        (SnowflakeConnection, Session)
    """
    print("=" * 80)
    print("CONNECTING TO SNOWFLAKE")
    print("=" * 80)
    
    print(f"\nConnecting using Snow CLI configuration: {connection_name}")
    
    try:
        connection = SnowflakeConnection.from_snow_cli(connection_name)
        session = connection.session
        
        # Get connection info
        current_account = session.sql("SELECT CURRENT_ACCOUNT()").collect()[0][0]
        current_user = session.sql("SELECT CURRENT_USER()").collect()[0][0]
        current_role = session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
        current_warehouse = session.sql("SELECT CURRENT_WAREHOUSE()").collect()[0][0]
        
        print(f"✓ Connected successfully!")
        print(f"\nConnection Details:")
        print(f"  Account:   {current_account}")
        print(f"  User:      {current_user}")
        print(f"  Role:      {current_role}")
        print(f"  Warehouse: {current_warehouse}")
        
        return connection, session
        
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure Snow CLI is installed: snow --version")
        print(f"  2. Check connection configuration: snow connection list")
        print(f"  3. Test connection: snow connection test -c {connection_name}")
        sys.exit(1)


def create_database_and_schema(session, database='ML_SHOWCASE', schema='MODELS'):
    """
    Create database and schema for Model Registry.
    
    Parameters:
    -----------
    session : Session
        Snowpark session
    database : str
        Database name
    schema : str
        Schema name
        
    Returns:
    --------
    tuple
        (database_name, schema_name)
    """
    print(f"\n{'=' * 80}")
    print("CREATING DATABASE AND SCHEMA")
    print("=" * 80)
    
    print(f"\nDatabase: {database}")
    print(f"Schema:   {schema}")
    
    # Create database
    print(f"\nCreating database {database}...")
    session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()
    print(f"✓ Database created/verified")
    
    # Use database
    session.sql(f"USE DATABASE {database}").collect()
    
    # Create schema
    print(f"Creating schema {schema}...")
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}").collect()
    print(f"✓ Schema created/verified")
    
    # Use schema
    session.sql(f"USE SCHEMA {schema}").collect()
    
    # Verify
    current_db = session.sql("SELECT CURRENT_DATABASE()").collect()[0][0]
    current_schema = session.sql("SELECT CURRENT_SCHEMA()").collect()[0][0]
    
    print(f"\nCurrent Context:")
    print(f"  Database: {current_db}")
    print(f"  Schema:   {current_schema}")
    
    return database, schema


def check_privileges(session, database, schema):
    """
    Check required privileges for Model Registry operations.
    
    Parameters:
    -----------
    session : Session
        Snowpark session
    database : str
        Database name
    schema : str
        Schema name
    """
    print(f"\n{'=' * 80}")
    print("CHECKING PRIVILEGES")
    print("=" * 80)
    
    print(f"\nRequired Privileges for Model Registry:")
    print(f"  • USAGE on database {database}")
    print(f"  • USAGE on schema {schema}")
    print(f"  • CREATE MODEL on schema {schema}")
    print(f"  • CREATE STAGE on schema {schema} (for model storage)")
    
    # Check current role
    current_role = session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
    print(f"\nCurrent Role: {current_role}")
    
    # Try to check grants (may fail if user doesn't have permission to see grants)
    try:
        print(f"\nAttempting to verify grants...")
        grants = session.sql(f"SHOW GRANTS ON SCHEMA {database}.{schema}").collect()
        print(f"✓ Found {len(grants)} grants on schema")
    except Exception as e:
        print(f"⚠ Could not verify grants (this is normal for some roles)")
        print(f"  If model operations fail, contact your Snowflake admin")
    
    print(f"\n✓ Privilege check complete")
    print(f"  If you can create models, you have sufficient privileges")


def initialize_registry(session, database, schema):
    """
    Initialize the Snowflake Model Registry.
    
    This is the core operation that creates a Registry object for managing models.
    
    Parameters:
    -----------
    session : Session
        Snowpark session
    database : str
        Database name
    schema : str
        Schema name
        
    Returns:
    --------
    Registry
        Initialized Model Registry object
    """
    print(f"\n{'=' * 80}")
    print("INITIALIZING MODEL REGISTRY")
    print("=" * 80)
    
    print(f"\nRegistry Parameters:")
    print(f"  session:       Snowpark session object")
    print(f"  database_name: {database}")
    print(f"  schema_name:   {schema}")
    
    print(f"\nCreating Registry object...")
    
    # Initialize the registry
    # PARAMETER DOCUMENTATION:
    # ------------------------
    # session (required):
    #   - Type: snowflake.snowpark.Session
    #   - Purpose: Active Snowpark session for database operations
    #   - Best Practice: Reuse session across operations
    #
    # database_name (required):
    #   - Type: str
    #   - Purpose: Database where models will be stored
    #   - Best Practice: Use dedicated database for ML models
    #   - Must be a valid Snowflake identifier
    #
    # schema_name (required):
    #   - Type: str
    #   - Purpose: Schema where models will be stored
    #   - Best Practice: Use dedicated schema (e.g., MODELS, REGISTRY)
    #   - Must be a valid Snowflake identifier
    #   - Any schema can be used as a registry (no initialization needed)
    
    registry = Registry(
        session=session,
        database_name=database,
        schema_name=schema
    )
    
    print(f"✓ Registry initialized successfully!")
    
    # Display registry information
    print(f"\nRegistry Information:")
    print(f"  Location: {registry.location}")
    
    return registry


def list_existing_models(registry):
    """
    List any existing models in the registry.
    
    Parameters:
    -----------
    registry : Registry
        Initialized Model Registry
    """
    print(f"\n{'=' * 80}")
    print("LISTING EXISTING MODELS")
    print("=" * 80)
    
    try:
        print(f"\nQuerying models in {registry.location}...")
        
        # Get list of models
        models_df = registry.show_models()
        
        if len(models_df) == 0:
            print(f"✓ No models found (registry is empty)")
            print(f"  This is expected for a new registry")
        else:
            print(f"✓ Found {len(models_df)} model(s):")
            print(f"\n{models_df}")
            
            # Display model details
            print(f"\nModel Summary:")
            for idx, row in models_df.iterrows():
                print(f"  {idx + 1}. {row['name']}")
                if 'default_version_name' in row:
                    print(f"     Default version: {row['default_version_name']}")
        
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
        print(f"  This may indicate the registry is empty")


def display_registry_capabilities(registry):
    """
    Display Model Registry capabilities and key methods.
    
    Parameters:
    -----------
    registry : Registry
        Initialized Model Registry
    """
    print(f"\n{'=' * 80}")
    print("MODEL REGISTRY CAPABILITIES")
    print("=" * 80)
    
    print(f"\nKey Methods:")
    print(f"\n1. log_model()")
    print(f"   Purpose: Register a new model or model version")
    print(f"   Usage: registry.log_model(model, model_name, version_name, ...)")
    print(f"   Returns: ModelVersion object")
    
    print(f"\n2. get_model()")
    print(f"   Purpose: Retrieve a model by name")
    print(f"   Usage: model = registry.get_model('MODEL_NAME')")
    print(f"   Returns: Model object")
    
    print(f"\n3. show_models()")
    print(f"   Purpose: List all models in the registry")
    print(f"   Usage: df = registry.show_models()")
    print(f"   Returns: pandas DataFrame")
    
    print(f"\n4. delete_model()")
    print(f"   Purpose: Delete a model and all its versions")
    print(f"   Usage: registry.delete_model('MODEL_NAME')")
    print(f"   Returns: None")
    
    print(f"\nModel Object Methods:")
    print(f"  • model.default - Get default version")
    print(f"  • model.version('v1') - Get specific version")
    print(f"  • model.versions() - List all versions")
    print(f"  • model.show_versions() - Show versions as DataFrame")
    
    print(f"\nModelVersion Methods:")
    print(f"  • version.run() - Run inference")
    print(f"  • version.load() - Load model to local Python")
    print(f"  • version.show_functions() - List available functions")
    print(f"  • version.show_metrics() - Display model metrics")
    
    print(f"\nSupported Model Types:")
    print(f"  • scikit-learn (sklearn)")
    print(f"  • XGBoost")
    print(f"  • LightGBM")
    print(f"  • PyTorch")
    print(f"  • TensorFlow")
    print(f"  • Hugging Face Transformers")
    print(f"  • MLflow PyFunc")
    print(f"  • Snowpark ML")
    print(f"  • Custom models (any pickleable Python object)")


def display_best_practices():
    """Display best practices for using Model Registry."""
    print(f"\n{'=' * 80}")
    print("BEST PRACTICES")
    print("=" * 80)
    
    print(f"\n1. Organization:")
    print(f"   • Use dedicated database for ML models (e.g., ML, ML_SHOWCASE)")
    print(f"   • Use dedicated schema for model registry (e.g., MODELS, REGISTRY)")
    print(f"   • Consider separate schemas for dev/staging/prod")
    
    print(f"\n2. Naming Conventions:")
    print(f"   • Model names: UPPERCASE_WITH_UNDERSCORES")
    print(f"   • Version names: v1, v2, v1_production, v2_20250110")
    print(f"   • Use descriptive names that indicate purpose")
    
    print(f"\n3. Versioning:")
    print(f"   • Always specify version_name for production models")
    print(f"   • Use semantic versioning or date-based versions")
    print(f"   • Set default version to the production model")
    print(f"   • Keep old versions for rollback capability")
    
    print(f"\n4. Metadata:")
    print(f"   • Always include metrics when logging models")
    print(f"   • Add comments describing the model and training data")
    print(f"   • Document hyperparameters in comments or metrics")
    
    print(f"\n5. Dependencies:")
    print(f"   • Specify exact package versions for reproducibility")
    print(f"   • Use conda_dependencies for packages in Snowflake channel")
    print(f"   • Test models in target environment before production")
    
    print(f"\n6. Security:")
    print(f"   • Use role-based access control (RBAC)")
    print(f"   • Grant minimal necessary privileges")
    print(f"   • Audit model access and usage")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("STEP 4: INITIALIZE SNOWFLAKE MODEL REGISTRY")
    print("=" * 80)
    print("\nThis script sets up the Snowflake Model Registry.")
    print("The registry is where we'll store and manage ML models.")
    
    # Configuration
    CONNECTION_NAME = 'legalzoom'
    DATABASE = 'ML_SHOWCASE'
    SCHEMA = 'MODELS'
    
    # Step 1: Connect to Snowflake
    connection, session = connect_to_snowflake(CONNECTION_NAME)
    
    # Step 2: Create database and schema
    db, schema = create_database_and_schema(session, DATABASE, SCHEMA)
    
    # Step 3: Check privileges
    check_privileges(session, db, schema)
    
    # Step 4: Initialize registry
    registry = initialize_registry(session, db, schema)
    
    # Step 5: List existing models
    list_existing_models(registry)
    
    # Step 6: Display capabilities
    display_registry_capabilities(registry)
    
    # Step 7: Display best practices
    display_best_practices()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Connected to Snowflake")
    print(f"✓ Database created: {db}")
    print(f"✓ Schema created: {schema}")
    print(f"✓ Registry initialized: {registry.location}")
    print(f"✓ Ready to log models!")
    
    # Close connection
    connection.close()
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print("Run the next script to load the .pkl file and log it to the registry:")
    print("  python 05_load_and_package.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

