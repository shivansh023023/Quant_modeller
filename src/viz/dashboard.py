"""
Dashboard creation utilities for Quant Lab.

This module provides functions to create comprehensive dashboards
for strategy analysis and performance visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Try to import plotly for interactive dashboards
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

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

from ..core.metrics import PerformanceMetrics, RiskMetrics


def create_strategy_dashboard(
    equity_curve: pd.Series,
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    title: str = "Strategy Dashboard",
    interactive: bool = True,
    save_path: Optional[str] = None
) -> Any:
    """
    Create a comprehensive strategy dashboard.
    
    Args:
        equity_curve: Series of portfolio values over time
        returns: Series of returns
        benchmark_returns: Optional benchmark returns for comparison
        trades: Optional DataFrame containing trade information
        feature_importance: Optional dictionary of feature importance scores
        title: Dashboard title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the dashboard
        
    Returns:
        Dashboard object (plotly figure or matplotlib figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _create_strategy_dashboard_plotly(
            equity_curve, returns, benchmark_returns, trades, feature_importance, title, save_path
        )
    else:
        return _create_strategy_dashboard_matplotlib(
            equity_curve, returns, benchmark_returns, trades, feature_importance, title, save_path
        )


def _create_strategy_dashboard_plotly(
    equity_curve: pd.Series,
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    trades: Optional[pd.DataFrame],
    feature_importance: Optional[Dict[str, float]],
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Create interactive plotly strategy dashboard."""
    
    # Calculate metrics
    performance_calc = PerformanceMetrics()
    risk_calc = RiskMetrics()
    
    performance_metrics = performance_calc.calculate_all_metrics(returns)
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Equity Curve', 'Drawdown',
            'Monthly Returns', 'Rolling Sharpe',
            'Returns Distribution', 'Feature Importance' if feature_importance else 'Risk Metrics'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name='Strategy', line=dict(width=3)),
        row=1, col=1
    )
    if benchmark_returns is not None:
        benchmark_curve = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
        fig.add_trace(
            go.Scatter(x=benchmark_curve.index, y=benchmark_curve.values, name='Benchmark', 
                      line=dict(width=3, dash='dash')),
            row=1, col=1
        )
    
    # 2. Drawdown
    drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonexty', name='Drawdown',
                  fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Monthly Returns Heatmap
    monthly_returns = returns.resample('M').sum()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(
        go.Heatmap(z=heatmap_data.values, x=month_labels, y=heatmap_data.index,
                   colorscale='RdYlGn', zmid=0, name='Monthly Returns'),
        row=2, col=1
    )
    
    # 4. Rolling Sharpe
    rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
    fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Rolling Sharpe'),
        row=2, col=2
    )
    
    # 5. Returns Distribution
    fig.add_trace(
        go.Histogram(x=returns.values, nbinsx=50, name='Returns', opacity=0.7),
        row=3, col=1
    )
    
    # 6. Feature Importance or Risk Metrics
    if feature_importance:
        # Sort feature importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]  # Top 10 features
        
        feature_names, importance_scores = zip(*top_features)
        fig.add_trace(
            go.Bar(x=importance_scores, y=feature_names, orientation='h', name='Feature Importance'),
            row=3, col=2
        )
    else:
        # Risk metrics summary
        risk_summary = pd.Series(risk_metrics)
        fig.add_trace(
            go.Bar(x=list(risk_summary.values), y=list(risk_summary.index), orientation='h', name='Risk Metrics'),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
    fig.update_xaxes(title_text="Returns", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_xaxes(title_text="Value", row=3, col=2)
    fig.update_yaxes(title_text="Metric", row=3, col=2)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def _create_strategy_dashboard_matplotlib(
    equity_curve: pd.Series,
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    trades: Optional[pd.DataFrame],
    feature_importance: Optional[Dict[str, float]],
    title: str,
    save_path: Optional[str]
) -> Any:
    """Create matplotlib strategy dashboard."""
    
    # Calculate metrics
    performance_calc = PerformanceMetrics()
    risk_calc = RiskMetrics()
    
    performance_metrics = performance_calc.calculate_all_metrics(returns)
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # 1. Equity Curve
    axes[0, 0].plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2)
    if benchmark_returns is not None:
        benchmark_curve = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
        axes[0, 0].plot(benchmark_curve.index, benchmark_curve.values, label='Benchmark', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Monthly Returns Heatmap
    monthly_returns = returns.resample('M').sum()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    im = axes[1, 0].imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('Monthly Returns Heatmap')
    axes[1, 0].set_xticks(range(12))
    axes[1, 0].set_xticklabels(month_labels)
    axes[1, 0].set_yticks(range(len(heatmap_data.index)))
    axes[1, 0].set_yticklabels(heatmap_data.index)
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Rolling Sharpe
    rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    axes[1, 1].set_title('Rolling Sharpe Ratio (252-day)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Returns Distribution
    axes[2, 0].hist(returns.values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[2, 0].set_title('Returns Distribution')
    axes[2, 0].set_xlabel('Returns')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Feature Importance or Risk Metrics
    if feature_importance:
        # Sort feature importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]  # Top 10 features
        
        feature_names, importance_scores = zip(*top_features)
        bars = axes[2, 1].barh(range(len(feature_names)), importance_scores, color='skyblue', alpha=0.7)
        axes[2, 1].set_title('Top 10 Feature Importance')
        axes[2, 1].set_xlabel('Importance Score')
        axes[2, 1].set_yticks(range(len(feature_names)))
        axes[2, 1].set_yticklabels(feature_names)
        axes[2, 1].grid(True, alpha=0.3, axis='x')
    else:
        # Risk metrics summary
        risk_summary = pd.Series(risk_metrics)
        bars = axes[2, 1].barh(range(len(risk_summary)), risk_summary.values, color='lightcoral', alpha=0.7)
        axes[2, 1].set_title('Risk Metrics Summary')
        axes[2, 1].set_xlabel('Value')
        axes[2, 1].set_yticks(range(len(risk_summary)))
        axes[2, 1].set_yticklabels(risk_summary.index)
        axes[2, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_performance_dashboard(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Performance Dashboard",
    interactive: bool = True,
    save_path: Optional[str] = None
) -> Any:
    """
    Create a performance-focused dashboard.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns for comparison
        title: Dashboard title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the dashboard
        
    Returns:
        Dashboard object (plotly figure or matplotlib figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _create_performance_dashboard_plotly(returns, benchmark_returns, title, save_path)
    else:
        return _create_performance_dashboard_matplotlib(returns, benchmark_returns, title, save_path)


def _create_performance_dashboard_plotly(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Create interactive plotly performance dashboard."""
    
    # Calculate performance metrics
    performance_calc = PerformanceMetrics()
    performance_metrics = performance_calc.calculate_all_metrics(returns)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns', 'Rolling Performance',
            'Monthly Returns Heatmap', 'Performance Metrics'
        )
    )
    
    # 1. Cumulative Returns
    cumulative_returns = (1 + returns).cumprod()
    fig.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name='Strategy', line=dict(width=3)),
        row=1, col=1
    )
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative.values, name='Benchmark', 
                      line=dict(width=3, dash='dash')),
            row=1, col=1
        )
    
    # 2. Rolling Performance
    rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Rolling Sharpe', line=dict(width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Rolling Volatility', line=dict(width=2), yaxis='y2'),
        row=1, col=2
    )
    
    # 3. Monthly Returns Heatmap
    monthly_returns = returns.resample('M').sum()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(
        go.Heatmap(z=heatmap_data.values, x=month_labels, y=heatmap_data.index,
                   colorscale='RdYlGn', zmid=0, name='Monthly Returns'),
        row=2, col=1
    )
    
    # 4. Performance Metrics Summary
    key_metrics = ['annualized_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio']
    metric_values = [performance_metrics.get(metric, 0) for metric in key_metrics]
    
    fig.add_trace(
        go.Bar(x=key_metrics, y=metric_values, name='Performance Metrics'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Volatility", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Metric", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def _create_performance_dashboard_matplotlib(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    title: str,
    save_path: Optional[str]
) -> Any:
    """Create matplotlib performance dashboard."""
    
    # Calculate performance metrics
    performance_calc = PerformanceMetrics()
    performance_metrics = performance_calc.calculate_all_metrics(returns)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # 1. Cumulative Returns
    cumulative_returns = (1 + returns).cumprod()
    axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='Strategy', linewidth=2)
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, label='Benchmark', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Rolling Performance
    rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)
    
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(rolling_sharpe.index, rolling_sharpe.values, 'b-', label='Rolling Sharpe', linewidth=2)
    line2 = ax2.plot(rolling_vol.index, rolling_vol.values, 'r-', label='Rolling Volatility', linewidth=2)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sharpe Ratio', color='b')
    ax2.set_ylabel('Volatility', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 3. Monthly Returns Heatmap
    monthly_returns = returns.resample('M').sum()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    im = axes[1, 0].imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('Monthly Returns Heatmap')
    axes[1, 0].set_xticks(range(12))
    axes[1, 0].set_xticklabels(month_labels)
    axes[1, 0].set_yticks(range(len(heatmap_data.index)))
    axes[1, 0].set_yticklabels(heatmap_data.index)
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Performance Metrics Summary
    key_metrics = ['annualized_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio']
    metric_values = [performance_metrics.get(metric, 0) for metric in key_metrics]
    
    bars = axes[1, 1].bar(range(len(key_metrics)), metric_values, color='skyblue', alpha=0.7)
    axes[1, 1].set_title('Performance Metrics Summary')
    axes[1, 1].set_xlabel('Metric')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(range(len(key_metrics)))
    axes[1, 1].set_xticklabels(key_metrics, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_risk_dashboard(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Risk Dashboard",
    interactive: bool = True,
    save_path: Optional[str] = None
) -> Any:
    """
    Create a risk-focused dashboard.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns for comparison
        title: Dashboard title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the dashboard
        
    Returns:
        Dashboard object (plotly figure or matplotlib figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _create_risk_dashboard_plotly(returns, benchmark_returns, title, save_path)
    else:
        return _create_risk_dashboard_matplotlib(returns, benchmark_returns, title, save_path)


def _create_risk_dashboard_plotly(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Create interactive plotly risk dashboard."""
    
    # Calculate risk metrics
    risk_calc = RiskMetrics()
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rolling Volatility', 'Value at Risk',
            'Returns Distribution', 'Risk Metrics Summary'
        )
    )
    
    # 1. Rolling Volatility
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Rolling Volatility', line=dict(width=3)),
        row=1, col=1
    )
    
    # 2. Value at Risk
    var_95 = returns.rolling(252).quantile(0.05)
    var_99 = returns.rolling(252).quantile(0.01)
    
    fig.add_trace(
        go.Scatter(x=var_95.index, y=var_95.values, name='VaR 95%', line=dict(width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=var_99.index, y=var_99.values, name='VaR 99%', line=dict(width=2)),
        row=1, col=2
    )
    
    # 3. Returns Distribution
    fig.add_trace(
        go.Histogram(x=returns.values, nbinsx=50, name='Returns', opacity=0.7),
        row=2, col=1
    )
    
    # 4. Risk Metrics Summary
    key_risk_metrics = ['volatility', 'var_95', 'cvar_95', 'downside_deviation', 'ulcer_index']
    risk_values = [risk_metrics.get(metric, 0) for metric in key_risk_metrics]
    
    fig.add_trace(
        go.Bar(x=key_risk_metrics, y=risk_values, name='Risk Metrics'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (Annualized)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Value at Risk", row=1, col=2)
    fig.update_xaxes(title_text="Returns", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Risk Metric", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def _create_risk_dashboard_matplotlib(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    title: str,
    save_path: Optional[str]
) -> Any:
    """Create matplotlib risk dashboard."""
    
    # Calculate risk metrics
    risk_calc = RiskMetrics()
    risk_metrics = risk_calc.calculate_all_risk_metrics(returns, benchmark_returns)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # 1. Rolling Volatility
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)
    axes[0, 0].plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='blue')
    axes[0, 0].set_title('Rolling Volatility (252-day)')
    axes[0, 0].set_ylabel('Volatility (Annualized)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Value at Risk
    var_95 = returns.rolling(252).quantile(0.05)
    var_99 = returns.rolling(252).quantile(0.01)
    
    axes[0, 1].plot(var_95.index, var_95.values, linewidth=2, label='VaR 95%', color='orange')
    axes[0, 1].plot(var_99.index, var_99.values, linewidth=2, label='VaR 99%', color='red')
    axes[0, 1].set_title('Value at Risk (252-day)')
    axes[0, 1].set_ylabel('VaR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Returns Distribution
    axes[1, 0].hist(returns.values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Returns')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Risk Metrics Summary
    key_risk_metrics = ['volatility', 'var_95', 'cvar_95', 'downside_deviation', 'ulcer_index']
    risk_values = [risk_metrics.get(metric, 0) for metric in key_risk_metrics]
    
    bars = axes[1, 1].bar(range(len(key_risk_metrics)), risk_values, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Risk Metrics Summary')
    axes[1, 1].set_xlabel('Risk Metric')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(range(len(key_risk_metrics)))
    axes[1, 1].set_xticklabels(key_risk_metrics, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
