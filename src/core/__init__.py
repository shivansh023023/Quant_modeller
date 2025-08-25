"""
Core utilities for data handling, metrics, and cross-validation.

This module contains the foundational components used across
the quant-lab system.
"""

from .data_api import DataAPI, DataCatalog
from .metrics import PerformanceMetrics, RiskMetrics
from .cross_validation import WalkForwardCV, PurgedKFoldCV, TimeSeriesSplit, CrossValidationManager
from .config import ConfigManager, get_config, get_api_key, is_service_enabled

__all__ = [
    "DataAPI",
    "DataCatalog", 
    "PerformanceMetrics",
    "RiskMetrics",
    "WalkForwardCV",
    "PurgedKFoldCV",
    "TimeSeriesSplit",
    "CrossValidationManager",
    "ConfigManager",
    "get_config",
    "get_api_key",
    "is_service_enabled",
]
