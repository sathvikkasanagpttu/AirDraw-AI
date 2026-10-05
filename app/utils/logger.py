"""
AirDraw AI - Logging Infrastructure
===================================
Production-grade structured logging with clear formatting and level control.
"""

import logging
import sys
from typing import Optional


class AirDrawFormatter(logging.Formatter):
    """Clean, human-readable console formatter with visual level badges."""

    COLORS = {
        logging.DEBUG: "\033[36m[DEBUG]\033[0m",
        logging.INFO: "\033[32m[INFO]\033[0m",
        logging.WARNING: "\033[33m[WARNING]\033[0m",
        logging.ERROR: "\033[31m[ERROR]\033[0m",
        logging.CRITICAL: "\033[41m[CRITICAL]\033[0m",
    }

    NO_COLOR = {
        logging.DEBUG: "[DEBUG]",
        logging.INFO: "[INFO]",
        logging.WARNING: "[WARNING]",
        logging.ERROR: "[ERROR]",
        logging.CRITICAL: "[CRITICAL]",
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        tag_map = self.COLORS if self.use_color else self.NO_COLOR
        tag = tag_map.get(record.levelno, f"[{record.levelname}]")
        timestamp = self.formatTime(record, "%H:%M:%S")
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
        return f"{tag} [{timestamp}] [{record.name}] {message}"


def get_logger(name: str = "AirDrawAI", level: int = logging.INFO) -> logging.Logger:
    """Obtain or configure a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AirDrawFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
