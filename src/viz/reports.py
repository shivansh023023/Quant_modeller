"""
Report generation utilities for Quant Lab.

This module provides functions to generate comprehensive reports
for strategy analysis and backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from ..core.metrics import PerformanceMetrics, RiskMetrics


def generate_performance_report(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    output_dir: str = "reports"
) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns for comparison
        strategy_name: Name of the strategy
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    performance_calc = PerformanceMetrics()
    risk_calc = RiskMetrics()
    
    performance_metrics = performance_calc.calculate_all_metrics(returns)
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"performance_report_{strategy_name}_{timestamp}.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Performance Report: {strategy_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Metrics\n\n")
        for metric, value in performance_metrics.items():
            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
        
        f.write("\n## Risk Metrics\n\n")
        for metric, value in risk_metrics.items():
            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
        
        f.write("\n## Summary Statistics\n\n")
        f.write(f"- **Total Return:** {(1 + returns).prod() - 1:.2%}\n")
        f.write(f"- **Annualized Return:** {returns.mean() * 252:.2%}\n")
        f.write(f"- **Volatility:** {returns.std() * np.sqrt(252):.2%}\n")
        f.write(f"- **Sharpe Ratio:** {performance_metrics.get('sharpe_ratio', 0):.2f}\n")
        f.write(f"- **Max Drawdown:** {performance_metrics.get('max_drawdown', 0):.2%}\n")
    
    return report_file


def generate_risk_report(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    output_dir: str = "reports"
) -> str:
    """
    Generate a comprehensive risk report.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns for comparison
        strategy_name: Name of the strategy
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate risk metrics
    risk_calc = RiskMetrics()
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"risk_report_{strategy_name}_{timestamp}.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Risk Report: {strategy_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Risk Metrics\n\n")
        for metric, value in risk_metrics.items():
            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
        
        f.write("\n## Risk Analysis\n\n")
        f.write(f"- **Volatility:** {returns.std() * np.sqrt(252):.2%}\n")
        f.write(f"- **VaR 95%:** {returns.quantile(0.05):.2%}\n")
        f.write(f"- **VaR 99%:** {returns.quantile(0.01):.2%}\n")
        f.write(f"- **CVaR 95%:** {returns[returns <= returns.quantile(0.05)].mean():.2%}\n")
        f.write(f"- **Skewness:** {returns.skew():.2f}\n")
        f.write(f"- **Kurtosis:** {returns.kurtosis():.2f}\n")
    
    return report_file


def generate_trade_analysis_report(
    trades: pd.DataFrame,
    strategy_name: str = "Strategy",
    output_dir: str = "reports"
) -> str:
    """
    Generate a trade analysis report.
    
    Args:
        trades: DataFrame containing trade information
        strategy_name: Name of the strategy
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if trades.empty:
        raise ValueError("No trades to analyze")
    
    # Calculate trade statistics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    losing_trades = len(trades[trades['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"trade_analysis_{strategy_name}_{timestamp}.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Trade Analysis Report: {strategy_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Trade Summary\n\n")
        f.write(f"- **Total Trades:** {total_trades}\n")
        f.write(f"- **Winning Trades:** {winning_trades}\n")
        f.write(f"- **Losing Trades:** {losing_trades}\n")
        f.write(f"- **Win Rate:** {win_rate:.2%}\n")
        
        f.write("\n## Trade Performance\n\n")
        f.write(f"- **Average Win:** ${avg_win:,.2f}\n")
        f.write(f"- **Average Loss:** ${avg_loss:,.2f}\n")
        f.write(f"- **Profit Factor:** {profit_factor:.2f}\n")
        f.write(f"- **Total PnL:** ${trades['pnl'].sum():,.2f}\n")
        f.write(f"- **Best Trade:** ${trades['pnl'].max():,.2f}\n")
        f.write(f"- **Worst Trade:** ${trades['pnl'].min():,.2f}\n")
    
    return report_file
