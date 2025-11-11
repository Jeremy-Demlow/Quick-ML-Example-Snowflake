# Snowflake ML Registry Showcase

> Complete end-to-end demonstration of Snowflake Model Registry with comprehensive parameter documentation and best practices.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Snowflake](https://img.shields.io/badge/Snowflake-ML%20Registry-blue.svg)](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)

## Overview

This showcase demonstrates the complete ML workflow with Snowflake Model Registry using **Jupyter notebooks** built with **nbdev**:

1. **[Generate Data](nbs/00_generate_data.ipynb)** - Create synthetic classification dataset
2. **[Train Model](nbs/01_train_model.ipynb)** - Train XGBoost classifier with hyperparameter tuning
3. **[Serialize Model](nbs/02_save_pickle.ipynb)** - Save model as .pkl file (pickle/joblib)
4. **[Create Registry](nbs/03_create_registry.ipynb)** - Initialize Snowflake Model Registry
5. **[Log Model](nbs/04_load_and_package.ipynb)** - Upload .pkl to registry with comprehensive parameters
6. **[Inference (Warehouse)](nbs/05_inference_warehouse.ipynb)** - Run predictions on Snowflake Warehouse
7. **[Inference (SPCS)](nbs/06_inference_spcs.ipynb)** - Deploy to Snowpark Container Services (optional)

## 🌟 Key Features

- **📚 Comprehensive Parameter Documentation**: Every `log_model()` parameter explained with purpose, use cases, and best practices
- **🚀 Both Deployment Options**: Warehouse (simple) and SPCS (advanced)
- **💻 Python & SQL APIs**: Examples of both inference methods
- **✅ Production-Ready**: Error handling, logging, best practices
- **📓 Interactive Notebooks**: Step-by-step Jupyter notebooks with nbdev

## Prerequisites

### Snowflake Account
- Snowflake account (any cloud provider)
- Virtual warehouse for compute
- Appropriate privileges (CREATE DATABASE, CREATE SCHEMA, CREATE MODEL)

### Software
- Python 3.10 or later
- Snow CLI installed and configured
- Conda for environment management

## Installation

### 1. Create Conda Environment

```bash
# Clone the repository
git clone https://github.com/jdemlow/Quick-ML-Example-Snowflake.git
cd Quick-ML-Example-Snowflake

# Create and activate conda environment
conda env create -f environment.yml
conda activate ml-registry-showcase
```

The environment includes all necessary packages:
- `snowflake-snowpark-python` - Snowflake Python API
- `snowflake-ml-python` - Snowflake ML library
- `xgboost`, `scikit-learn` - ML libraries
- `jupyter`, `nbdev` - Notebook development
- `pandas`, `numpy`, `matplotlib`, `seaborn` - Data science tools

### 2. Configure Snowflake Connection

```bash
# Add Snowflake connection using Snow CLI
snow connection add

# Follow the prompts to enter:
# - Connection name (e.g., 'legalzoom')
# - Account identifier
# - Username
# - Password or authentication method
# - Role
# - Warehouse

# Test the connection
snow connection test -c legalzoom
```

### 3. Set Up Snowflake Privileges

Ensure your Snowflake role has the required privileges:

```sql
-- Required privileges
GRANT CREATE DATABASE ON ACCOUNT TO ROLE <your_role>;
GRANT CREATE SCHEMA ON DATABASE <database> TO ROLE <your_role>;
GRANT CREATE MODEL ON SCHEMA <database>.<schema> TO ROLE <your_role>;
GRANT USAGE ON WAREHOUSE <warehouse> TO ROLE <your_role>;
```

## Quick Start

### Option 1: Run Jupyter Notebooks (Recommended)

```bash
# Activate the environment
conda activate ml-registry-showcase

# Start Jupyter
jupyter notebook

# Open nbs/index.ipynb to get started
```

Follow the notebooks in order:

1. **[00_generate_data.ipynb](nbs/00_generate_data.ipynb)** - Generate synthetic data
2. **[01_train_model.ipynb](nbs/01_train_model.ipynb)** - Train XGBoost model
3. **[02_save_pickle.ipynb](nbs/02_save_pickle.ipynb)** - Serialize model
4. **[03_create_registry.ipynb](nbs/03_create_registry.ipynb)** - Setup registry
5. **[04_load_and_package.ipynb](nbs/04_load_and_package.ipynb)** - Log model ⭐ **KEY STEP**
6. **[05_inference_warehouse.ipynb](nbs/05_inference_warehouse.ipynb)** - Run inference
7. **[06_inference_spcs.ipynb](nbs/06_inference_spcs.ipynb)** - SPCS deployment (optional)

### Option 2: Run Python Scripts

The original Python scripts are available in `model_registry_showcase/`:

```bash
cd model_registry_showcase

# Run scripts in order
python 01_generate_data.py
python 02_train_model.py
python 03_save_pickle.py
python 04_create_registry.py
python 05_load_and_package.py
python 06a_inference_warehouse.py
python 06b_inference_spcs.py  # Optional
```

## Project Structure

```
Quick-ML-Example-Snowflake/
├── README.md                          # This file
├── environment.yml                    # Conda environment specification
├── settings.ini                       # nbdev configuration
├── setup.py                          # Python package setup
│
├── nbs/                              # 📓 Jupyter Notebooks
│   ├── index.ipynb                   # Main documentation
│   ├── 00_generate_data.ipynb        # Data generation
│   ├── 01_train_model.ipynb          # Model training
│   ├── 02_save_pickle.ipynb          # Model serialization
│   ├── 03_create_registry.ipynb      # Registry setup
│   ├── 04_load_and_package.ipynb     # Model logging ⭐
│   ├── 05_inference_warehouse.ipynb  # Warehouse inference
│   └── 06_inference_spcs.ipynb       # SPCS inference
│
└── model_registry_showcase/          # 🐍 Original Python Scripts
    ├── 01_generate_data.py
    ├── 02_train_model.py
    ├── 03_save_pickle.py
    ├── 04_create_registry.py
    ├── 05_load_and_package.py
    ├── 06a_inference_warehouse.py
    ├── 06b_inference_spcs.py
    ├── connections.py                # Snowflake connection helper
    └── requirements.txt              # Python dependencies
```

## Comprehensive Parameter Reference

### registry.log_model() Parameters

The most important function in Model Registry. See **[04_load_and_package.ipynb](nbs/04_load_and_package.ipynb)** for complete documentation.

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

#### Optional Parameters - Deployment

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_platforms` | list[str] | ["WAREHOUSE"] | Where to run: "WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES", or both |

For complete parameter documentation, see **[04_load_and_package.ipynb](nbs/04_load_and_package.ipynb)**.

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

### 3. Dependency Management

```python
# Pin exact versions for production
conda_dependencies = [
    "xgboost==2.0.0",
    "scikit-learn==1.3.0",
    "pandas==2.0.0"
]
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

### Environment Issues

```bash
# Recreate conda environment
conda env remove -n ml-registry-showcase
conda env create -f environment.yml
conda activate ml-registry-showcase
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

### nbdev

- [nbdev Documentation](https://nbdev.fast.ai/)
- [nbdev Tutorial](https://nbdev.fast.ai/tutorials/tutorial.html)

## Contributing

This is a showcase project for educational purposes. Feel free to fork and adapt for your own use cases.

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

Made with ❤️ using [nbdev](https://nbdev.fast.ai/)

