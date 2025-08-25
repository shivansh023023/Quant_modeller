"""
Feature engineering and management module.

This module provides tools for creating, managing, and applying
features to financial data for quantitative trading strategies.
"""

from .registry import FeatureRegistry
from .generators import FeatureGenerator
from .utils import FeatureUtils

__all__ = [
    "FeatureRegistry",
    "FeatureGenerator", 
    "FeatureUtils",
]
