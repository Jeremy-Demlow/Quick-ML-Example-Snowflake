import logging
import os
import sys
from typing import Optional

_LOGGER_CONFIGURED = False

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
ENV_VAR_LOG_LEVEL = "MODEL_REGISTRY_LOG_LEVEL"


def _normalize_level(value: str, fallback: int) -> int:
    """Resolve a logging level string or numeric value to an int."""
    if value is None:
        return fallback

    if isinstance(value, str):
        level = logging.getLevelName(value.upper())
        if isinstance(level, int):
            return level
    elif isinstance(value, int):
        return value

    return fallback


def setup_logging(
    default_level: int = DEFAULT_LOG_LEVEL,
    env_var: str = ENV_VAR_LOG_LEVEL,
    force: bool = False,
) -> None:
    """Configure root logging once for scripts and notebooks."""
    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED and not force:
        return

    env_level = os.getenv(env_var)
    level = _normalize_level(env_level, default_level)

    root_logger = logging.getLogger()

    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
        root_logger.addHandler(handler)

    root_logger.setLevel(level)
    _LOGGER_CONFIGURED = True


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger for the given name."""
    setup_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def log_section(
    logger: logging.Logger,
    title: str,
    level: int = logging.INFO,
    width: int = 80,
    pad_char: str = "=",
) -> None:
    """Log a section heading similar to the previous banner prints."""
    border = pad_char * width if pad_char else ""
    if border:
        logger.log(level, border)
    logger.log(level, title)
    if border:
        logger.log(level, border)


def set_global_level(level: int) -> None:
    """Allow callers to adjust root log level at runtime."""
    setup_logging(force=False)
    logging.getLogger().setLevel(level)

