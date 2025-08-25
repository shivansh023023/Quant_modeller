"""
Backtesting engine for quantitative trading strategies.

This module provides comprehensive backtesting capabilities with
realistic assumptions and transaction costs.
"""

from .engine import BacktestEngine
from .result import BacktestResult
from .slippage import SlippageModel
from .position_sizing import PositionSizer

__all__ = [
    "BacktestEngine",
    "BacktestResult", 
    "SlippageModel",
    "PositionSizer",
]
