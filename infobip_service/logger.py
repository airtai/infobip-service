# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/Logger.ipynb (unless otherwise specified).

__all__ = [
    "supress_timestamps",
    "get_default_logger_configuration",
    "should_supress_timestamps",
    "get_logger",
    "logger_spaces_added",
    "set_level",
]

# Cell

import logging

# Internal Cell
import logging.config
from typing import Any

# Cell

# Logger Levels
# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0

should_supress_timestamps: bool = False


def supress_timestamps(flag: bool = True) -> None:
    """Supress logger timestamp.

    Args:
        flag: If not set, then the default value **True** will be used to supress the timestamp
            from the logger messages
    """
    global should_supress_timestamps
    should_supress_timestamps = flag


def get_default_logger_configuration(level: int = logging.INFO) -> dict[str, Any]:
    """Return the common configurations for the logger.

    Args:
        level: Logger level to set

    Returns:
        A dict with default logger configuration

    """
    global should_supress_timestamps

    if should_supress_timestamps:
        format = "[%(levelname)s] %(name)s: %(message)s"
    else:
        format = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"

    date_fmt = "%y-%m-%d %H:%M:%S"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": format, "datefmt": date_fmt},
        },
        "handlers": {
            "default": {
                "level": level,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
        },
        "loggers": {
            "": {"handlers": ["default"], "level": level},  # root logger
        },
    }
    return logging_config


# Cell

logger_spaces_added: list[str] = []


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return the logger class with default logging configuration.

    Args:
        name: Pass the __name__ variable as name while calling
        level: Used to configure logging, default value `logging.INFO` logs
            info messages and up.

    Returns:
        The logging.Logger class with default/custom logging configuration

    """
    config = get_default_logger_configuration(level=level)
    logging.config.dictConfig(config)

    logger = logging.getLogger(name)

    return logger


# Cell


def set_level(level: int) -> None:
    """Set logger level.

    Args:
        level: Logger level to set
    """
    # Getting all loggers that has either airt or __main__ in the name
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if ("airt" in name) or ("__main__" in name)
    ]

    for logger in loggers:
        logger.setLevel(level)
