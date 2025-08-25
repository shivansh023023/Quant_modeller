"""
Performance and risk metrics calculation.

This module provides comprehensive calculation of trading strategy
performance metrics including returns, risk measures, and ratios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class PerformanceMetrics:
    """Calculate performance metrics for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.annual_factor = 252  # Trading days per year
    
    def calculate_returns(
        self, 
        prices: pd.Series, 
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            Series of returns
        """
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:  # simple
            returns = prices.pct_change()
        
        return returns
    
    def calculate_cumulative_returns(
        self, 
        returns: pd.Series, 
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate cumulative returns.
        
        Args:
            returns: Return series
            method: Cumulative method ('log' or 'simple')
            
        Returns:
            Series of cumulative returns
        """
        if method == "log":
            cumulative = returns.cumsum()
        else:  # simple
            cumulative = (1 + returns).cumprod() - 1
        
        return cumulative
    
    def calculate_annualized_return(
        self, 
        returns: pd.Series
    ) -> float:
        """
        Calculate annualized return.
        
        Args:
            returns: Return series
            
        Returns:
            Annualized return
        """
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.annual_factor
        annualized = (1 + total_return) ** (1 / years) - 1
        
        return annualized
    
    def calculate_volatility(
        self, 
        returns: pd.Series, 
        annualize: bool = True
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Return series
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility
        """
        volatility = returns.std()
        
        if annualize:
            volatility *= np.sqrt(self.annual_factor)
        
        return volatility
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (uses instance default if None)
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / self.annual_factor
        sharpe = excess_returns.mean() / returns.std()
        
        if returns.std() == 0:
            return 0.0
        
        return sharpe * np.sqrt(self.annual_factor)
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: Optional[float] = None,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (uses instance default if None)
            target_return: Target return for downside deviation
            
        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / self.annual_factor
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return np.inf
        
        sortino = excess_returns.mean() / downside_deviation
        return sortino * np.sqrt(self.annual_factor)
    
    def calculate_max_drawdown(
        self, 
        cumulative_returns: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        # Find start date (peak before max drawdown)
        peak_value = running_max.loc[end_date]
        start_date = cumulative_returns[cumulative_returns == peak_value].index[0]
        
        return max_dd, start_date, end_date
    
    def calculate_calmar_ratio(
        self, 
        returns: pd.Series
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Return series
            
        Returns:
            Calmar ratio
        """
        annualized_return = self.calculate_annualized_return(returns)
        cumulative_returns = self.calculate_cumulative_returns(returns)
        max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns)
        
        if max_dd == 0:
            return np.inf
        
        return abs(annualized_return / max_dd)
    
    def calculate_win_rate(
        self, 
        returns: pd.Series
    ) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Return series
            
        Returns:
            Win rate as decimal
        """
        return (returns > 0).mean()
    
    def calculate_profit_factor(
        self, 
        returns: pd.Series
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            returns: Return series
            
        Returns:
            Profit factor
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def calculate_turnover(
        self, 
        positions: pd.DataFrame,
        method: str = "absolute"
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            positions: DataFrame with position weights over time
            method: Turnover calculation method ('absolute' or 'net')
            
        Returns:
            Annualized turnover rate
        """
        if method == "absolute":
            # Absolute turnover: sum of absolute changes
            turnover = positions.diff().abs().sum(axis=1)
        else:  # net
            # Net turnover: sum of positive changes
            turnover = positions.diff().clip(lower=0).sum(axis=1)
        
        # Annualize
        annual_turnover = turnover.mean() * self.annual_factor
        
        return annual_turnover
    
    def calculate_hit_rate(
        self, 
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate hit rate (percentage of returns above threshold).
        
        Args:
            returns: Return series
            threshold: Threshold for counting hits
            
        Returns:
            Hit rate as decimal
        """
        return (returns > threshold).mean()
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness."""
        return stats.skew(returns)
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate return kurtosis."""
        return stats.kurtosis(returns)
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR at specified confidence level
        """
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
            
        Returns:
            CVaR at specified confidence level
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_all_metrics(
        self, 
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Return series
            positions: Optional position DataFrame for turnover calculation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = self.calculate_annualized_return(returns)
        metrics['volatility'] = self.calculate_volatility(returns)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        
        # Drawdown metrics
        cumulative_returns = self.calculate_cumulative_returns(returns)
        max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns)
        metrics['max_drawdown'] = max_dd
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        
        # Win/loss metrics
        metrics['win_rate'] = self.calculate_win_rate(returns)
        metrics['profit_factor'] = self.calculate_profit_factor(returns)
        metrics['hit_rate'] = self.calculate_hit_rate(returns)
        
        # Distribution metrics
        metrics['skewness'] = self.calculate_skewness(returns)
        metrics['kurtosis'] = self.calculate_kurtosis(returns)
        
        # Risk metrics
        metrics['var_95'] = self.calculate_var(returns, 0.05)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.05)
        
        # Turnover (if positions provided)
        if positions is not None:
            metrics['turnover'] = self.calculate_turnover(positions)
        
        return metrics


class RiskMetrics:
    """Calculate risk metrics for trading strategies."""
    
    def __init__(self):
        """Initialize risk metrics calculator."""
        pass
    
    def calculate_beta(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate beta relative to benchmark.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Beta coefficient
        """
        # Align series
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        if benchmark_variance == 0:
            return np.nan
        
        beta = covariance / benchmark_variance
        return beta
    
    def calculate_tracking_error(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate tracking error.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Tracking error
        """
        # Align series
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate excess returns
        excess_returns = strategy_aligned - benchmark_aligned
        
        # Calculate tracking error
        tracking_error = excess_returns.std()
        
        return tracking_error
    
    def calculate_information_ratio(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate information ratio.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Information ratio
        """
        # Align series
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate excess returns
        excess_returns = strategy_aligned - benchmark_aligned
        
        # Calculate information ratio
        if excess_returns.std() == 0:
            return np.nan
        
        information_ratio = excess_returns.mean() / excess_returns.std()
        
        return information_ratio
    
    def calculate_correlation(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate correlation with benchmark.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Correlation coefficient
        """
        # Align series
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate correlation
        correlation = strategy_aligned.corr(benchmark_aligned)
        
        return correlation
    
    def calculate_downside_deviation(
        self, 
        returns: pd.Series, 
        target_return: float = 0.0
    ) -> float:
        """
        Calculate downside deviation.
        
        Args:
            returns: Return series
            target_return: Target return for downside calculation
            
        Returns:
            Downside deviation
        """
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = downside_returns.std()
        
        return downside_deviation
    
    def calculate_ulcer_index(
        self, 
        cumulative_returns: pd.Series
    ) -> float:
        """
        Calculate Ulcer Index.
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Ulcer Index
        """
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        
        return ulcer_index
    
    def calculate_gain_to_pain_ratio(
        self, 
        returns: pd.Series
    ) -> float:
        """
        Calculate gain-to-pain ratio.
        
        Args:
            returns: Return series
            
        Returns:
            Gain-to-pain ratio
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf
        
        return gains / losses
    
    def calculate_all_risk_metrics(
        self, 
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all risk metrics.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Optional benchmark return series
            
        Returns:
            Dictionary of all risk metrics
        """
        risk_metrics = {}
        
        # Volatility metrics
        risk_metrics['volatility'] = strategy_returns.std()
        risk_metrics['downside_deviation'] = self.calculate_downside_deviation(strategy_returns)
        
        # Drawdown metrics
        cumulative_returns = (1 + strategy_returns).cumprod()
        risk_metrics['ulcer_index'] = self.calculate_ulcer_index(cumulative_returns)
        
        # Win/loss metrics
        risk_metrics['gain_to_pain_ratio'] = self.calculate_gain_to_pain_ratio(strategy_returns)
        
        # Benchmark-relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            risk_metrics['beta'] = self.calculate_beta(strategy_returns, benchmark_returns)
            risk_metrics['tracking_error'] = self.calculate_tracking_error(strategy_returns, benchmark_returns)
            risk_metrics['information_ratio'] = self.calculate_information_ratio(strategy_returns, benchmark_returns)
            risk_metrics['correlation'] = self.calculate_correlation(strategy_returns, benchmark_returns)
        
        return risk_metrics
