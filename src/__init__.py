"""
Quant Lab: AI-Powered Quantitative Trading Strategy Development Tool

A comprehensive research workbench for systematic trading ideas.
"""

__version__ = "0.1.0"
__author__ = "Quant Lab Team"
__email__ = "team@quantlab.ai"

from .strategies import StrategySpec
from .backtest import BacktestResult, BacktestEngine
from .core.metrics import PerformanceMetrics, RiskMetrics
from .core.config import ConfigManager, get_config, get_api_key
from .ai import IdeaGenerator
from .tracking import MLflowTracker, ExperimentManager, MetricsLogger
from .viz import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns,
    plot_rolling_metrics,
    plot_feature_importance,
    plot_correlation_matrix,
    plot_returns_distribution,
    plot_rolling_volatility,
    create_strategy_dashboard,
    create_performance_dashboard,
    create_risk_dashboard,
    generate_performance_report,
    generate_risk_report,
    generate_trade_analysis_report
)

__all__ = [
    "StrategySpec",
    "BacktestResult",
    "BacktestEngine",
    "PerformanceMetrics",
    "RiskMetrics",
    "ConfigManager",
    "get_config",
    "get_api_key",
    "IdeaGenerator",
    "MLflowTracker",
    "ExperimentManager",
    "MetricsLogger",
    # Visualization functions
    "plot_equity_curve",
    "plot_drawdown",
    "plot_monthly_returns",
    "plot_rolling_metrics",
    "plot_feature_importance",
    "plot_correlation_matrix",
    "plot_returns_distribution",
    "plot_rolling_volatility",
    # Dashboard functions
    "create_strategy_dashboard",
    "create_performance_dashboard",
    "create_risk_dashboard",
    # Report functions
    "generate_performance_report",
    "generate_risk_report",
    "generate_trade_analysis_report",
]
