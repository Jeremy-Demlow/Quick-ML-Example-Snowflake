# Snowflake Model Registry Showcase

Complete end-to-end demonstration of Snowflake Model Registry with comprehensive parameter documentation and best practices.

## Overview

This showcase demonstrates the complete ML workflow with Snowflake Model Registry:

1. **Generate Data** - Create synthetic classification dataset
2. **Train Model** - Train XGBoost classifier with hyperparameter tuning
3. **Serialize Model** - Save model as .pkl file (pickle/joblib)
4. **Create Registry** - Initialize Snowflake Model Registry
5. **Log Model** - Upload .pkl to registry with comprehensive parameters
6. **Inference (Warehouse)** - Run predictions on Snowflake Warehouse
7. **Inference (SPCS)** - Deploy to Snowpark Container Services (optional)

## Key Features

- **Comprehensive Parameter Documentation**: Every `log_model()` parameter explained with purpose, use cases, and best practices
- **Both Deployment Options**: Warehouse (simple) and SPCS (advanced)
- **Python & SQL APIs**: Examples of both inference methods
- **Production-Ready**: Error handling, logging, best practices
- **Step-by-Step**: Each script is self-contained and well-documented

## Prerequisites

### Snowflake Account
- Snowflake account (any cloud provider)
- Virtual warehouse for compute
- Appropriate privileges (CREATE DATABASE, CREATE SCHEMA, CREATE MODEL)

### Software
- Python 3.10 or later
- Snow CLI installed and configured
- Conda or venv for environment management

### Snowflake Privileges
```sql
-- Required privileges
GRANT CREATE DATABASE ON ACCOUNT TO ROLE <your_role>;
GRANT CREATE SCHEMA ON DATABASE <database> TO ROLE <your_role>;
GRANT CREATE MODEL ON SCHEMA <database>.<schema> TO ROLE <your_role>;
GRANT USAGE ON WAREHOUSE <warehouse> TO ROLE <your_role>;
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the showcase directory
cd model_registry_showcase

# Create conda environment
conda create -n model-registry-showcase python=3.10
conda activate model-registry-showcase

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Snowflake Connection

```bash
# Add Snowflake connection using Snow CLI
snow connection add

# Test connection
snow connection test -c legalzoom
```

### 3. Run the Workflow

Execute scripts in order:

```bash
# Step 1: Generate synthetic data
python 01_generate_data.py

# Step 2: Train XGBoost model
python 02_train_model.py

# Step 3: Serialize model to .pkl
python 03_save_pickle.py

# Step 4: Initialize Model Registry
python 04_create_registry.py

# Step 5: Load .pkl and log to registry (KEY STEP)
python 05_load_and_package.py

# Step 6a: Run inference on Warehouse
python 06a_inference_warehouse.py

# Step 6b: Run inference on SPCS (optional)
python 06b_inference_spcs.py
```

## File Descriptions

### Core Scripts

| Script | Description | Key Concepts |
|--------|-------------|--------------|
| `01_generate_data.py` | Generate synthetic classification data | sklearn.datasets, Snowflake table upload |
| `02_train_model.py` | Train XGBoost with hyperparameter tuning | XGBClassifier, GridSearchCV, metrics |
| `03_save_pickle.py` | Serialize model (pickle vs joblib) | Model serialization, file formats |
| `04_create_registry.py` | Initialize Model Registry | Registry(), database/schema setup |
| `05_load_and_package.py` | **Log model with ALL parameters** | **registry.log_model() - comprehensive** |
| `06a_inference_warehouse.py` | Inference on Warehouse | Python API, SQL API, performance |
| `06b_inference_spcs.py` | Inference on SPCS | Compute pools, containerization |

### Supporting Files

| File | Description |
|------|-------------|
| `connections.py` | Snowflake connection helper |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Comprehensive Parameter Reference

### registry.log_model() Parameters

The most important function in Model Registry. See `05_load_and_package.py` for complete documentation.

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | object | Trained model (must be pickleable) |
| `model_name` | str | Unique model identifier (Snowflake identifier) |

#### Optional Parameters - Versioning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version_name` | str | Auto-generated | Version identifier (e.g., "v1", "v_20250110") |

#### Optional Parameters - Metadata

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comment` | str | None | Human-readable description |
| `metrics` | dict | None | Performance metrics (max 100 KB) |

#### Optional Parameters - Dependencies

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conda_dependencies` | list[str] | None | Conda packages (e.g., ["xgboost==2.0.0"]) |
| `pip_requirements` | list[str] | None | PyPI packages (SPCS only) |
| `python_version` | str | Current | Python version ("3.8", "3.9", "3.10", "3.11") |

#### Optional Parameters - Signature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_input_data` | DataFrame | None | Sample data for signature inference (recommended) |
| `signatures` | dict | None | Manual signature specification (advanced) |

#### Optional Parameters - Deployment

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_platforms` | list[str] | ["WAREHOUSE"] | Where to run: "WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES", or both |

#### Optional Parameters - Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `options` | dict | {} | Advanced configuration (see below) |

**options Dictionary:**

```python
options = {
    "enable_explainability": False,      # Enable SHAP explanations
    "relax_version": True,               # Allow flexible dependency versions
    "embed_local_ml_library": False,     # Use Snowflake's snowflake-ml-python
    "target_methods": ["predict", "predict_proba"],  # Methods to expose
    "method_options": {                  # Per-method configuration
        "predict": {"case_sensitive": True}
    }
}
```

#### Optional Parameters - Code and Files

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code_paths` | list[str] | [] | Custom Python code directories |
| `ext_modules` | list | [] | External modules to pickle |
| `user_files` | list[str] | [] | Additional files (configs, images) |

## Warehouse vs SPCS Comparison

| Feature | Warehouse | SPCS |
|---------|-----------|------|
| **Model Size** | < 100 MB (optimal) | Any size |
| **Compute** | CPU only | CPU or GPU |
| **Dependencies** | Snowflake Conda channel | Any PyPI/Conda |
| **Setup** | Simple | Moderate |
| **Cold Start** | Seconds | Minutes (first time) |
| **Scaling** | Warehouse scaling | Container auto-scaling |
| **Cost** | Warehouse credits | Compute pool credits |
| **Best For** | Batch inference, small models | Real-time, large models, GPU |
| **External Access** | SQL/Python only | REST API available |

## Best Practices

### 1. Model Organization

```
ML_SHOWCASE (Database)
└── MODELS (Schema)
    ├── XGBOOST_CLASSIFIER
    │   ├── v1
    │   ├── v2
    │   └── v1_production (DEFAULT)
    └── OTHER_MODELS...
```

### 2. Naming Conventions

- **Model names**: `UPPERCASE_WITH_UNDERSCORES`
- **Version names**: `v1`, `v2`, `v1_production`, `v_20250110`
- **Database/Schema**: `ML_SHOWCASE`, `MODELS`

### 3. Versioning Strategy

```python
# Development
version_name = f"v_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Staging
version_name = "v1_staging"

# Production
version_name = "v1_production"

# Set as default
model.set_default_version("v1_production")
```

### 4. Dependency Management

```python
# Pin exact versions for production
conda_dependencies = [
    "xgboost==2.0.0",
    "scikit-learn==1.3.0",
    "pandas==2.0.0"
]

# Flexible versions for development
conda_dependencies = [
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
    "pandas"
]
```

### 5. Metrics Tracking

```python
metrics = {
    # Performance metrics
    'accuracy': 0.95,
    'f1_score': 0.94,
    'precision': 0.93,
    'recall': 0.96,
    'roc_auc': 0.97,
    
    # Model metadata
    'model_type': 'XGBClassifier',
    'training_date': '2025-01-10',
    'training_samples': 10000,
    'feature_count': 20,
    
    # Data metadata
    'data_source': 'synthetic_classification',
    'data_version': 'v1'
}
```

## Inference Examples

### Python API

```python
from snowflake.ml.registry import Registry

# Initialize registry
registry = Registry(session, database_name="ML_SHOWCASE", schema_name="MODELS")

# Get model
model = registry.get_model("XGBOOST_CLASSIFIER")
model_version = model.default

# Run inference
predictions = model_version.run(test_data, function_name="predict")
probabilities = model_version.run(test_data, function_name="predict_proba")
```

### SQL API

```sql
-- Use default version
SELECT 
    ID,
    XGBOOST_CLASSIFIER!predict(*) AS PREDICTION,
    XGBOOST_CLASSIFIER!predict_proba(*) AS PROBABILITIES
FROM ML_SHOWCASE.DATA.SYNTHETIC_DATA
LIMIT 100;

-- Use specific version
WITH model_v1 AS MODEL XGBOOST_CLASSIFIER VERSION "v1_production"
SELECT 
    ID,
    model_v1!predict(*) AS PREDICTION
FROM ML_SHOWCASE.DATA.SYNTHETIC_DATA;
```

## Troubleshooting

### Connection Issues

```bash
# Check Snow CLI configuration
snow connection list

# Test connection
snow connection test -c legalzoom

# Reconfigure if needed
snow connection add
```

### Permission Errors

```sql
-- Check current role
SELECT CURRENT_ROLE();

-- Check grants
SHOW GRANTS TO ROLE <your_role>;

-- Request privileges from admin
GRANT CREATE MODEL ON SCHEMA ML_SHOWCASE.MODELS TO ROLE <your_role>;
```

### Model Loading Errors

```python
# Check if model exists
models_df = registry.show_models()
print(models_df)

# Check model versions
model = registry.get_model("XGBOOST_CLASSIFIER")
versions_df = model.show_versions()
print(versions_df)

# Load specific version
model_version = model.version("v1")
```

### Dependency Errors

```python
# Use relax_version for development
options = {"relax_version": True}

# Pin exact versions for production
conda_dependencies = ["xgboost==2.0.0"]

# Check available packages
# Visit: https://repo.anaconda.com/pkgs/snowflake/
```

## Additional Resources

### Snowflake Documentation

- [Model Registry Overview](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)
- [Logging Models](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/logging-models)
- [Model Inference](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/inference)
- [SPCS Model Serving](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/spcs-model-serving)

### Snowflake ML Python

- [snowflake-ml-python Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index)
- [API Reference](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/index)

### Community

- [Snowflake Community](https://community.snowflake.com/)
- [Snowflake GitHub](https://github.com/Snowflake-Labs)

## License

This showcase is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Snowflake documentation
3. Contact your Snowflake account team
4. Post in Snowflake Community forums

---

**Happy Model Registry Exploration!** 🚀

