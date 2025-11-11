from setuptools import setup, find_packages

setup(
    name='snowflake_ml_registry_showcase',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'snowflake-snowpark-python>=1.11.0',
        'snowflake-ml-python>=1.8.0',
        'xgboost>=2.0.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'joblib>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
    ],
    python_requires='>=3.10',
    author='jdemlow',
    author_email='demlow@gmail.com',
    description='Complete end-to-end demonstration of Snowflake Model Registry',
    keywords='snowflake machine-learning model-registry xgboost',
)

