"""
Visualization and plotting utilities for Quant Lab.

This module provides comprehensive visualization capabilities
for strategy analysis, backtesting results, and performance metrics.
"""

from .charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns,
    plot_rolling_metrics,
    plot_feature_importance,
    plot_correlation_matrix,
    plot_returns_distribution,
    plot_rolling_volatility
)

from .dashboard import (
    create_strategy_dashboard,
    create_performance_dashboard,
    create_risk_dashboard
)

from .reports import (
    generate_performance_report,
    generate_risk_report,
    generate_trade_analysis_report
)

__all__ = [
    # Charts
    "plot_equity_curve",
    "plot_drawdown", 
    "plot_monthly_returns",
    "plot_rolling_metrics",
    "plot_feature_importance",
    "plot_correlation_matrix",
    "plot_returns_distribution",
    "plot_rolling_volatility",
    
    # Dashboards
    "create_strategy_dashboard",
    "create_performance_dashboard",
    "create_risk_dashboard",
    
    # Reports
    "generate_performance_report",
    "generate_risk_report",
    "generate_trade_analysis_report",
]
