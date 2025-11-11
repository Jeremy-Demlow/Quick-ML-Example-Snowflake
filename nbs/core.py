from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from connections import SnowflakeConnection
from custom_model import train_model_with_preprocessing
from logging_utils import get_logger, log_section

try:
    from snowflake.ml.registry import Registry
except ImportError:  # pragma: no cover - optional dependency during local dev
    Registry = None  # type: ignore

try:
    from snowflake.ml.model import TargetPlatform
except ImportError:  # pragma: no cover - optional dependency
    TargetPlatform = None  # type: ignore


logger = get_logger(__name__)


@dataclass
class DataConfig:
    n_samples: int = 10000
    n_features: int = 20
    random_state: int = 42
    csv_path: Path = Path("synthetic_data.csv")
    upload_to_snowflake: bool = True
    connection_name: str = "legalzoom"
    database: str = "ML_SHOWCASE"
    data_schema: str = "DATA"
    table_name: str = "SYNTHETIC_DATA"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    scaler_path: Path = Path("scaler.pkl")
    model_path: Path = Path("model.pkl")
    test_data_path: Path = Path("test_data.csv")
    metrics_path: Path = Path("model_metrics.json")


@dataclass
class RegistryConfig:
    connection_name: str = "legalzoom"
    database: str = "ML_SHOWCASE"
    schema: str = "MODELS"
    model_name: str = "LINEAR_REGRESSION_CUSTOM"
    user_files: Dict[str, list[str]] = field(default_factory=lambda: {"preprocessing": ["scaler.pkl"]})
    conda_dependencies: list[str] = field(
        default_factory=lambda: [
            "snowflake::scikit-learn==1.3.0",
            "snowflake::pandas==2.0.3",
            "snowflake::numpy==1.24.3",
        ]
    )
    python_version: str = "3.10"
    enable_explainability: bool = False
    target_platform_mode: str = "WAREHOUSE_ONLY"


@dataclass
class PipelineSteps:
    generate_data: bool = True
    train_model: bool = True
    verify_pickles: bool = True
    log_model: bool = True


@dataclass
class ServingConfig:
    enabled: bool = False
    compute_pool: Optional[str] = None
    service_name: Optional[str] = None
    min_instances: int = 1
    max_instances: int = 1
    instance_family: str = "CPU_X64_M"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    steps: PipelineSteps = field(default_factory=PipelineSteps)
    serving: ServingConfig = field(default_factory=ServingConfig)


def generate_synthetic_data(config: DataConfig) -> pd.DataFrame:
    """Generate synthetic regression data."""
    log_section(logger, "GENERATING SYNTHETIC DATASET")

    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=min(config.n_features, 15),
        n_targets=1,
        noise=10.0,
        bias=50.0,
        random_state=config.random_state,
    )

    feature_names = [f"FEATURE_{i:02d}" for i in range(config.n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["TARGET"] = y
    df.insert(0, "ID", range(1, len(df) + 1))

    logger.info("Dataset summary: samples=%s, features=%s", f"{config.n_samples:,}", config.n_features)
    logger.info("Target mean=%.2f std=%.2f", df["TARGET"].mean(), df["TARGET"].std())
    return df


def save_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Persist dataframe to CSV."""
    log_section(logger, "SAVING DATA TO CSV")
    df.to_csv(path, index=False)
    logger.info("Saved data to %s (%.2f MB)", path, path.stat().st_size / (1024 * 1024))
    return path


def upload_to_snowflake(df: pd.DataFrame, config: DataConfig) -> Optional[str]:
    """Upload dataframe to Snowflake if requested."""
    if not config.upload_to_snowflake:
        logger.info("Snowflake upload skipped (upload_to_snowflake=False)")
        return None

    log_section(logger, "UPLOADING DATA TO SNOWFLAKE")
    connection = SnowflakeConnection.from_snow_cli(config.connection_name)
    session = connection.session
    try:
        session.sql(f"CREATE DATABASE IF NOT EXISTS {config.database}").collect()
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {config.data_schema}").collect()
        session.sql(f"USE DATABASE {config.database}").collect()
        session.sql(f"USE SCHEMA {config.data_schema}").collect()

        session.create_dataframe(df).write.mode("overwrite").save_as_table(config.table_name)
        logger.info(
            "Uploaded data to %s.%s.%s", config.database, config.data_schema, config.table_name
        )
        return f"{config.database}.{config.data_schema}.{config.table_name}"
    finally:
        connection.close()


def split_training_data(
    df: pd.DataFrame, config: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train/test folds."""
    features = [col for col in df.columns if col.startswith("FEATURE_")]
    X = df[features]
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    logger.info(
        "Split data: train=%s, test=%s (test_size=%.0f%%)",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        config.test_size * 100,
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(
    scaler,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Compute evaluation metrics for train/test sets."""
    log_section(logger, "EVALUATING MODEL")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "train": {
            "mse": mean_squared_error(y_train, train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "mae": mean_absolute_error(y_train, train_pred),
            "r2": r2_score(y_train, train_pred),
        },
        "test": {
            "mse": mean_squared_error(y_test, test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "mae": mean_absolute_error(y_test, test_pred),
            "r2": r2_score(y_test, test_pred),
        },
    }

    for split in ("train", "test"):
        logger.info(
            "%s metrics: RMSE=%.4f MAE=%.4f R2=%.4f",
            split.capitalize(),
            metrics[split]["rmse"],
            metrics[split]["mae"],
            metrics[split]["r2"],
        )
    return metrics


def save_training_artifacts(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, Dict[str, float]],
    config: TrainConfig,
) -> None:
    """Persist test data and metrics for downstream steps."""
    log_section(logger, "SAVING TRAINING ARTIFACTS")
    test_df = X_test.copy()
    test_df["TARGET"] = y_test
    test_df.to_csv(config.test_data_path, index=False)
    logger.info("Stored test data at %s", config.test_data_path)

    with config.metrics_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Stored metrics at %s", config.metrics_path)


def load_pickle(path: Path):
    """Utility to load a pickle file."""
    with path.open("rb") as fh:
        return pickle.load(fh)


def verify_pickles(
    scaler_path: Path,
    model_path: Path,
    test_data_path: Path,
    sample_size: int = 10,
) -> bool:
    """Verify scaler and model pickles operate together."""
    log_section(logger, "VERIFYING PICKLE ARTIFACTS")
    if not scaler_path.exists() or not model_path.exists():
        logger.error("Missing pickles. scaler=%s exists=%s model=%s exists=%s",
                     scaler_path, scaler_path.exists(), model_path, model_path.exists())
        return False

    scaler = load_pickle(scaler_path)
    model = load_pickle(model_path)

    test_df = pd.read_csv(test_data_path)
    feature_cols = [col for col in test_df.columns if col.startswith("FEATURE_")]
    X_sample = test_df[feature_cols].head(sample_size)
    y_sample = test_df["TARGET"].head(sample_size) if "TARGET" in test_df else None

    predictions = model.predict(scaler.transform(X_sample))
    logger.info("Verified prediction pipeline on %s samples", len(X_sample))
    if y_sample is not None:
        mse = mean_squared_error(y_sample, predictions)
        logger.info("Sample MSE=%.4f", mse)
    return True


def init_registry(registry_cfg: RegistryConfig) -> Tuple[SnowflakeConnection, Registry]:
    """Initialize Snowflake Model Registry connection."""
    if Registry is None:
        raise RuntimeError("snowflake-ml-python is required to use the registry workflows.")

    log_section(logger, "INITIALIZING SNOWFLAKE REGISTRY")
    connection = SnowflakeConnection.from_snow_cli(registry_cfg.connection_name)
    session = connection.session
    registry = Registry(
        session=session,
        database_name=registry_cfg.database,
        schema_name=registry_cfg.schema,
    )
    logger.info("Registry ready at %s", registry.location)
    return connection, registry


def log_model_version(
    registry: Registry,
    registry_cfg: RegistryConfig,
    sample_data: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Any:
    """Log trained model into Snowflake registry with reference files."""
    log_section(logger, "LOGGING MODEL VERSION")
    model = load_pickle(Path("model.pkl"))
    scaler = load_pickle(Path("scaler.pkl"))

    for subdir, files in registry_cfg.user_files.items():
        for file_name in files:
            path = Path(file_name)
            if not path.exists():
                raise FileNotFoundError(f"user_files entry not found: {path}")
            if path.stat().st_size > 5 * 1024 * 1024 * 1024:
                raise ValueError(f"user_files entry exceeds 5GB limit: {path}")

    sample_scaled = pd.DataFrame(
        scaler.transform(sample_data),
        columns=sample_data.columns,
    )
    version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Logging model %s version %s", registry_cfg.model_name, version_name)

    target_platform = None
    if TargetPlatform and registry_cfg.target_platform_mode:
        target_platform = getattr(TargetPlatform, registry_cfg.target_platform_mode, None)
        if target_platform is None:
            logger.warning("Unknown target_platform_mode '%s'", registry_cfg.target_platform_mode)

    log_kwargs = dict(
        model=model,
        model_name=registry_cfg.model_name,
        version_name=version_name,
        comment=(
            "Linear Regression with custom preprocessing. "
            f"Logged on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        ),
        metrics=metrics,
        conda_dependencies=registry_cfg.conda_dependencies,
        python_version=registry_cfg.python_version,
        sample_input_data=sample_scaled,
        options={
            "enable_explainability": registry_cfg.enable_explainability,
            "relax_version": True,
            "target_methods": ["predict"],
        },
        user_files=registry_cfg.user_files,
    )

    if target_platform is not None:
        log_kwargs["target_platform"] = target_platform

    model_version = registry.log_model(**log_kwargs)
    logger.info("Logged model version: %s", model_version.version_name)
    return model_version


def run_pipeline(pipeline_cfg: PipelineConfig) -> Dict[str, Any]:
    """Execute selected pipeline steps."""
    outputs: Dict[str, Any] = {}

    if pipeline_cfg.steps.generate_data:
        df = generate_synthetic_data(pipeline_cfg.data)
        csv_path = save_to_csv(df, pipeline_cfg.data.csv_path)
        table_name = upload_to_snowflake(df, pipeline_cfg.data)
        outputs["dataframe"] = df
        outputs["csv_path"] = csv_path
        outputs["table_name"] = table_name
    else:
        logger.info("Generate data step skipped")
        df = pd.read_csv(pipeline_cfg.data.csv_path)

    if pipeline_cfg.steps.train_model:
        X_train, X_test, y_train, y_test = split_training_data(df, pipeline_cfg.train)
        scaler, model = train_model_with_preprocessing(X_train, y_train)
        metrics = evaluate_model(scaler, model, X_train, X_test, y_train, y_test)
        save_training_artifacts(X_test, y_test, metrics, pipeline_cfg.train)
        outputs["metrics"] = metrics
    else:
        logger.info("Train model step skipped")
        with pipeline_cfg.train.metrics_path.open() as fh:
            outputs["metrics"] = json.load(fh)

    if pipeline_cfg.steps.verify_pickles:
        verified = verify_pickles(
            pipeline_cfg.train.scaler_path,
            pipeline_cfg.train.model_path,
            pipeline_cfg.train.test_data_path,
        )
        outputs["verification_passed"] = verified

    if pipeline_cfg.steps.log_model:
        sample_df = pd.read_csv(pipeline_cfg.train.test_data_path)
        feature_cols = [col for col in sample_df.columns if col.startswith("FEATURE_")]
        sample_data = sample_df[feature_cols].head(5)
        connection, registry = init_registry(pipeline_cfg.registry)
        try:
            model_version = log_model_version(
                registry,
                pipeline_cfg.registry,
                sample_data,
                outputs.get("metrics", {}),
            )
            outputs["model_version"] = model_version.version_name
        finally:
            connection.close()

    if pipeline_cfg.serving.enabled:
        try:
            service = deploy_inference_service(pipeline_cfg.registry, pipeline_cfg.serving)
            outputs["service"] = service
        except Exception as exc:  # pragma: no cover - optional feature
            logger.warning("Service deployment skipped: %s", exc)

    return outputs


def pipeline_config_from_mapping(mapping: Dict[str, Any]) -> PipelineConfig:
    """Create a pipeline configuration from a nested mapping."""
    cfg = PipelineConfig()

    def _apply(target: Any, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if hasattr(target, key):
                current = getattr(target, key)
                if isinstance(current, Path) and not isinstance(value, Path):
                    setattr(target, key, Path(value))
                else:
                    setattr(target, key, value)

    if "data" in mapping:
        _apply(cfg.data, mapping["data"])
    if "train" in mapping:
        _apply(cfg.train, mapping["train"])
    if "registry" in mapping:
        _apply(cfg.registry, mapping["registry"])
    if "steps" in mapping:
        _apply(cfg.steps, mapping["steps"])
    if "serving" in mapping:
        _apply(cfg.serving, mapping["serving"])

    return cfg


def ensure_compute_pool(session, serving_cfg: ServingConfig) -> None:
    if not serving_cfg.compute_pool:
        return

    create_sql = f"""
    CREATE COMPUTE POOL IF NOT EXISTS {serving_cfg.compute_pool}
        MIN_NODES = {serving_cfg.min_instances}
        MAX_NODES = {serving_cfg.max_instances}
        INSTANCE_FAMILY = '{serving_cfg.instance_family}'
    """
    session.sql(create_sql).collect()
    logger.info("Compute pool ensured: %s", serving_cfg.compute_pool)


def deploy_inference_service(registry_cfg: RegistryConfig, serving_cfg: ServingConfig) -> Optional[Any]:
    if not serving_cfg.enabled:
        logger.info("SPCS deployment disabled by configuration")
        return None
    if Registry is None:
        raise RuntimeError("snowflake-ml-python is required for service deployment.")

    connection, registry = init_registry(registry_cfg)
    try:
        session = connection.session
        ensure_compute_pool(session, serving_cfg)

        model = registry.get_model(registry_cfg.model_name)
        model_version = model.default or (model.versions()[-1] if model.versions() else None)
        if model_version is None:
            raise RuntimeError("No model versions available to deploy.")

        deploy_kwargs = {}
        if serving_cfg.compute_pool:
            deploy_kwargs["compute_pool"] = serving_cfg.compute_pool
        if serving_cfg.service_name:
            deploy_kwargs["service_name"] = serving_cfg.service_name
        if hasattr(serving_cfg, "min_instances"):
            deploy_kwargs["min_instances"] = serving_cfg.min_instances
        if hasattr(serving_cfg, "max_instances"):
            deploy_kwargs["max_instances"] = serving_cfg.max_instances

        service = None
        if hasattr(model_version, "deploy_to_snowpark_container_services"):
            service = model_version.deploy_to_snowpark_container_services(**deploy_kwargs)
        elif hasattr(model_version, "deploy"):
            deploy_kwargs.setdefault("target_platform", "SNOWPARK_CONTAINER_SERVICES")
            service = model_version.deploy(**deploy_kwargs)
        else:
            raise AttributeError("Snowflake ML SDK does not expose an SPCS deployment helper in this version.")

        logger.info("Requested service deployment for %s", serving_cfg.service_name or model_version.version_name)
        return service
    finally:
        connection.close()

