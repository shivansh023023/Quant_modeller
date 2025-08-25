"""
Strategy definition and management module.

This module contains the core StrategySpec schema and related utilities
for defining quantitative trading strategies.
"""

from .schema import StrategySpec, StrategyBuilder
from .validator import StrategyValidator

__all__ = ["StrategySpec", "StrategyBuilder", "StrategyValidator"]
