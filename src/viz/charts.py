"""
Chart and plotting functions for Quant Lab.

This module provides various plotting functions for visualizing
financial data, strategy performance, and analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import warnings

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot equity curve with optional benchmark comparison.
    
    Args:
        equity_curve: Series of portfolio values over time
        benchmark_curve: Optional benchmark values for comparison
        title: Plot title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_equity_curve_plotly(equity_curve, benchmark_curve, title, save_path)
    else:
        return _plot_equity_curve_matplotlib(equity_curve, benchmark_curve, title, save_path, figsize)


def _plot_equity_curve_matplotlib(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series],
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of equity curve plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_curve is not None:
        ax.plot(benchmark_curve.index, benchmark_curve.values, label='Benchmark', linewidth=2, alpha=0.7)
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add value annotations
    if len(equity_curve) > 0:
        final_value = equity_curve.iloc[-1]
        ax.annotate(f'Final: ${final_value:,.0f}', 
                   xy=(equity_curve.index[-1], final_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_equity_curve_plotly(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series],
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of equity curve plot."""
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(width=3, color='blue')
    ))
    
    # Add benchmark if provided
    if benchmark_curve is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve.values,
            mode='lines',
            name='Benchmark',
            line=dict(width=3, color='red', dash='dash')
        ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_drawdown(
    drawdown: pd.Series,
    title: str = "Drawdown Analysis",
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot drawdown over time.
    
    Args:
        drawdown: Series of drawdown values over time
        title: Plot title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_drawdown_plotly(drawdown, title, save_path)
    else:
        return _plot_drawdown_matplotlib(drawdown, title, save_path, figsize)


def _plot_drawdown_matplotlib(
    drawdown: pd.Series,
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of drawdown plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add max drawdown annotation
    if len(drawdown) > 0:
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.annotate(f'Max DD: {max_dd:.1f}%', 
                   xy=(max_dd_date, max_dd),
                   xytext=(10, -10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                   fontsize=10, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_drawdown_plotly(
    drawdown: pd.Series,
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of drawdown plot."""
    fig = go.Figure()
    
    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red'),
        name='Drawdown'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot monthly returns heatmap.
    
    Args:
        returns: Series of returns
        title: Plot title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_monthly_returns_plotly(returns, title, save_path)
    else:
        return _plot_monthly_returns_matplotlib(returns, title, save_path, figsize)


def _plot_monthly_returns_matplotlib(
    returns: pd.Series,
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of monthly returns heatmap."""
    # Resample to monthly returns
    monthly_returns = returns.resample('M').sum()
    
    # Create year-month matrix
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    # Pivot to create heatmap data
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    
    # Customize axes
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Monthly Return', rotation=270, labelpad=20)
    
    # Add value annotations
    for i in range(len(heatmap_data.index)):
        for j in range(12):
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value):
                color = 'white' if abs(value) > 0.05 else 'black'
                ax.text(j, i, f'{value:.1%}', ha='center', va='center', 
                       color=color, fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_monthly_returns_plotly(
    returns: pd.Series,
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of monthly returns heatmap."""
    # Resample to monthly returns
    monthly_returns = returns.resample('M').sum()
    
    # Create year-month matrix
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    # Pivot to create heatmap data
    heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=returns.name or 'returns')
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=month_labels,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{val:.1%}' if not pd.isna(val) else '' for val in row] for row in heatmap_data.values],
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Year',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: List[str] = None,
    title: str = "Rolling Metrics",
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot rolling metrics over time.
    
    Args:
        returns: Series of returns
        window: Rolling window size (default: 252 for annual)
        metrics: List of metrics to plot (default: ['sharpe', 'volatility'])
        title: Plot title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if metrics is None:
        metrics = ['sharpe', 'volatility']
    
    if interactive and PLOTLY_AVAILABLE:
        return _plot_rolling_metrics_plotly(returns, window, metrics, title, save_path)
    else:
        return _plot_rolling_metrics_matplotlib(returns, window, metrics, title, save_path, figsize)


def _plot_rolling_metrics_matplotlib(
    returns: pd.Series,
    window: int,
    metrics: List[str],
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of rolling metrics plot."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(figsize[0], figsize[1] * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric == 'sharpe':
            rolling_metric = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            ylabel = 'Sharpe Ratio'
        elif metric == 'volatility':
            rolling_metric = returns.rolling(window).std() * np.sqrt(252)
            ylabel = 'Volatility (Annualized)'
        elif metric == 'returns':
            rolling_metric = returns.rolling(window).mean() * 252
            ylabel = 'Returns (Annualized)'
        else:
            continue
        
        axes[i].plot(rolling_metric.index, rolling_metric.values, linewidth=2)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        if i == len(metrics) - 1:
            axes[i].set_xlabel('Date', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_rolling_metrics_plotly(
    returns: pd.Series,
    window: int,
    metrics: List[str],
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of rolling metrics plot."""
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics)
    
    for i, metric in enumerate(metrics):
        if metric == 'sharpe':
            rolling_metric = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            ylabel = 'Sharpe Ratio'
        elif metric == 'volatility':
            rolling_metric = returns.rolling(window).std() * np.sqrt(252)
            ylabel = 'Volatility (Annualized)'
        elif metric == 'returns':
            rolling_metric = returns.rolling(window).mean() * 252
            ylabel = 'Returns (Annualized)'
        else:
            continue
        
        fig.add_trace(
            go.Scatter(x=rolling_metric.index, y=rolling_metric.values, name=metric.title()),
            row=i+1, col=1
        )
        
        fig.update_yaxes(title_text=ylabel, row=i+1, col=1)
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=300 * len(metrics)
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    title: str = "Feature Importance",
    top_n: Optional[int] = None,
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Any:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        title: Plot title
        top_n: Number of top features to show (default: all)
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_feature_importance_plotly(feature_names, importance_scores, title, top_n, save_path)
    else:
        return _plot_feature_importance_matplotlib(feature_names, importance_scores, title, top_n, save_path, figsize)


def _plot_feature_importance_matplotlib(
    feature_names: List[str],
    importance_scores: List[float],
    title: str,
    top_n: Optional[int],
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of feature importance plot."""
    # Sort by importance
    sorted_data = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_data = sorted_data[:top_n]
    
    features, scores = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(features)), scores, color='skyblue', alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_feature_importance_plotly(
    feature_names: List[str],
    importance_scores: List[float],
    title: str,
    top_n: Optional[int],
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of feature importance plot."""
    # Sort by importance
    sorted_data = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_data = sorted_data[:top_n]
    
    features, scores = zip(*sorted_data)
    
    fig = go.Figure(data=go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker_color='skyblue',
        text=[f'{score:.3f}' for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    method: str = 'pearson',
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Any:
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame containing features
        title: Plot title
        method: Correlation method ('pearson', 'spearman', 'kendall')
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)
    
    if interactive and PLOTLY_AVAILABLE:
        return _plot_correlation_matrix_plotly(corr_matrix, title, save_path)
    else:
        return _plot_correlation_matrix_matplotlib(corr_matrix, title, save_path, figsize)


def _plot_correlation_matrix_matplotlib(
    corr_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of correlation matrix plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Customize axes
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_correlation_matrix_plotly(
    corr_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of correlation matrix plot."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=[[f'{val:.2f}' for val in row] for row in corr_matrix.values],
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    bins: int = 50,
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Any:
    """
    Plot returns distribution histogram with normal distribution overlay.
    
    Args:
        returns: Series of returns
        title: Plot title
        bins: Number of histogram bins
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_returns_distribution_plotly(returns, title, bins, save_path)
    else:
        return _plot_returns_distribution_matplotlib(returns, title, bins, save_path, figsize)


def _plot_returns_distribution_matplotlib(
    returns: pd.Series,
    title: str,
    bins: int,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of returns distribution plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(returns.values, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_dist, 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkew: {returns.skew():.3f}\nKurt: {returns.kurtosis():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_returns_distribution_plotly(
    returns: pd.Series,
    title: str,
    bins: int,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of returns distribution plot."""
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=bins,
        name='Returns',
        opacity=0.7,
        marker_color='skyblue'
    ))
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=normal_dist,
        mode='lines',
        name=f'Normal (μ={mu:.3f}, σ={sigma:.3f})',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Returns',
        yaxis_title='Density',
        template='plotly_white',
        barmode='overlay'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    title: str = "Rolling Volatility",
    interactive: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot rolling volatility over time.
    
    Args:
        returns: Series of returns
        window: Rolling window size (default: 252 for annual)
        title: Plot title
        interactive: Whether to use interactive plotly plots
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib plots
        
    Returns:
        Plot object (matplotlib figure or plotly figure)
    """
    if interactive and PLOTLY_AVAILABLE:
        return _plot_rolling_volatility_plotly(returns, window, title, save_path)
    else:
        return _plot_rolling_volatility_matplotlib(returns, window, title, save_path, figsize)


def _plot_rolling_volatility_matplotlib(
    returns: pd.Series,
    window: int,
    title: str,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Matplotlib implementation of rolling volatility plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    
    # Plot rolling volatility
    ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='blue', label=f'{window}-day Rolling Vol')
    
    # Add mean volatility line
    mean_vol = rolling_vol.mean()
    ax.axhline(y=mean_vol, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_vol:.1%}')
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (Annualized)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add volatility statistics
    stats_text = f'Current: {rolling_vol.iloc[-1]:.1%}\nMin: {rolling_vol.min():.1%}\nMax: {rolling_vol.max():.1%}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_rolling_volatility_plotly(
    returns: pd.Series,
    window: int,
    title: str,
    save_path: Optional[str]
) -> go.Figure:
    """Plotly implementation of rolling volatility plot."""
    fig = go.Figure()
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    
    # Add rolling volatility
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        name=f'{window}-day Rolling Vol',
        line=dict(width=3, color='blue')
    ))
    
    # Add mean volatility line
    mean_vol = rolling_vol.mean()
    fig.add_hline(y=mean_vol, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_vol:.1%}")
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Volatility (Annualized)',
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig
