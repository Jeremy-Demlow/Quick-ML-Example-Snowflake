# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Create the Conda Environment

```bash
# Navigate to the project directory
cd /Users/jdemlow/github/Quick-ML-Example-Snowflake

# Create the conda environment (this takes 2-3 minutes)
conda env create -f environment.yml

# Activate the environment
conda activate ml-registry-showcase
```

### Step 2: Configure Snowflake Connection

```bash
# Add your Snowflake connection
snow connection add

# When prompted, enter:
# - Connection name: legalzoom (or your preference)
# - Account: your_account.region
# - User: your_username
# - Password: your_password
# - Role: ACCOUNTADMIN (or appropriate role)
# - Warehouse: COMPUTE_WH (or your warehouse)

# Test the connection
snow connection test -c legalzoom
```

### Step 3: Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook
```

### Step 4: Run the Notebooks

Open and run the notebooks in this order:

1. **`nbs/index.ipynb`** - Start here for an overview
2. **`nbs/00_generate_data.ipynb`** - Generate synthetic data (2 min)
3. **`nbs/01_train_model.ipynb`** - Train XGBoost model (3 min)
4. **`nbs/02_save_pickle.ipynb`** - Serialize model (1 min)
5. **`nbs/03_create_registry.ipynb`** - Initialize registry (1 min)
6. **`nbs/04_load_and_package.ipynb`** - Log model to registry ⭐ **KEY** (2 min)
7. **`nbs/05_inference_warehouse.ipynb`** - Run inference (2 min)
8. **`nbs/06_inference_spcs.ipynb`** - SPCS deployment (optional) (5 min)

**Total Time**: ~15 minutes for notebooks 1-7

## 📝 What You'll Learn

- ✅ How to create and manage Snowflake Model Registry
- ✅ Complete model lifecycle: train → serialize → register → deploy
- ✅ Comprehensive `registry.log_model()` parameters
- ✅ Python API vs SQL API for inference
- ✅ Warehouse vs SPCS deployment options
- ✅ Production best practices

## 🎯 Key Notebook: 04_load_and_package.ipynb

This notebook contains **comprehensive documentation** of every `registry.log_model()` parameter:
- Required parameters
- Optional parameters for versioning, metadata, dependencies
- Deployment targets (Warehouse vs SPCS)
- Advanced options
- Code and file management

## 💡 Tips

1. **Run notebooks in order** - Each builds on the previous one
2. **Check your Snowflake privileges** - You need CREATE DATABASE, CREATE SCHEMA, CREATE MODEL
3. **Use Warehouse for first run** - It's simpler than SPCS
4. **Read the comments** - Each notebook has detailed explanations
5. **Experiment** - Try changing parameters and see what happens!

## 🐛 Common Issues

### "Connection failed"
```bash
# Verify your Snow CLI configuration
snow connection list
snow connection test -c legalzoom
```

### "Permission denied"
```sql
-- Check your role in Snowflake
SELECT CURRENT_ROLE();

-- Request privileges from admin
GRANT CREATE MODEL ON SCHEMA ML_SHOWCASE.MODELS TO ROLE <your_role>;
```

### "Package not found"
```bash
# Recreate the environment
conda env remove -n ml-registry-showcase
conda env create -f environment.yml
conda activate ml-registry-showcase
```

## 📚 More Help

- See [README.md](README.md) for full documentation
- Check [Snowflake ML Registry docs](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)
- Review individual notebook markdown cells for detailed explanations

---

**Ready? Let's build!** 🚀

```bash
conda activate ml-registry-showcase
jupyter notebook
```

