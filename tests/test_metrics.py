"""
Tests for performance and risk metrics calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.metrics import PerformanceMetrics, RiskMetrics


class TestPerformanceMetrics:
    """Test performance metrics calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        return returns
    
    @pytest.fixture
    def performance_calc(self):
        """Create PerformanceMetrics instance."""
        return PerformanceMetrics(risk_free_rate=0.02)
    
    def test_annualized_return_calculation(self, performance_calc, sample_returns):
        """Test annualized return calculation."""
        ann_return = performance_calc.calculate_annualized_return(sample_returns)
        assert isinstance(ann_return, float)
        assert -1.0 < ann_return < 10.0  # Reasonable bounds
    
    def test_volatility_calculation(self, performance_calc, sample_returns):
        """Test volatility calculation."""
        volatility = performance_calc.calculate_volatility(sample_returns)
        assert isinstance(volatility, float)
        assert volatility > 0
    
    def test_sharpe_ratio_calculation(self, performance_calc, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = performance_calc.calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert -5.0 < sharpe < 5.0  # Reasonable bounds
    
    def test_sortino_ratio_calculation(self, performance_calc, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = performance_calc.calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_max_drawdown_calculation(self, performance_calc, sample_returns):
        """Test maximum drawdown calculation."""
        cumulative_returns = performance_calc.calculate_cumulative_returns(sample_returns)
        max_dd, start_date, end_date = performance_calc.calculate_max_drawdown(cumulative_returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)
        assert start_date <= end_date
    
    def test_calmar_ratio_calculation(self, performance_calc, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = performance_calc.calculate_calmar_ratio(sample_returns)
        assert isinstance(calmar, float)
        assert calmar >= 0  # Should be positive
    
    def test_win_rate_calculation(self, performance_calc, sample_returns):
        """Test win rate calculation."""
        win_rate = performance_calc.calculate_win_rate(sample_returns)
        assert isinstance(win_rate, float)
        assert 0.0 <= win_rate <= 1.0
    
    def test_profit_factor_calculation(self, performance_calc, sample_returns):
        """Test profit factor calculation."""
        profit_factor = performance_calc.calculate_profit_factor(sample_returns)
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0
    
    def test_var_calculation(self, performance_calc, sample_returns):
        """Test VaR calculation."""
        var_95 = performance_calc.calculate_var(sample_returns, 0.05)
        assert isinstance(var_95, float)
        assert var_95 <= 0  # VaR should be negative for losses
    
    def test_cvar_calculation(self, performance_calc, sample_returns):
        """Test CVaR calculation."""
        cvar_95 = performance_calc.calculate_cvar(sample_returns, 0.05)
        assert isinstance(cvar_95, float)
        assert cvar_95 <= 0  # CVaR should be negative for losses
    
    def test_all_metrics_calculation(self, performance_calc, sample_returns):
        """Test calculating all metrics at once."""
        metrics = performance_calc.calculate_all_metrics(sample_returns)
        
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'win_rate', 'profit_factor',
            'hit_rate', 'skewness', 'kurtosis', 'var_95', 'cvar_95'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric]) or metric in ['skewness', 'kurtosis']


class TestRiskMetrics:
    """Test risk metrics calculations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        strategy_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
        
        return strategy_returns, benchmark_returns
    
    @pytest.fixture
    def risk_calc(self):
        """Create RiskMetrics instance."""
        return RiskMetrics()
    
    def test_beta_calculation(self, risk_calc, sample_data):
        """Test beta calculation."""
        strategy_returns, benchmark_returns = sample_data
        beta = risk_calc.calculate_beta(strategy_returns, benchmark_returns)
        
        assert isinstance(beta, float)
        assert not np.isnan(beta)
        assert -5.0 < beta < 5.0  # Reasonable bounds
    
    def test_tracking_error_calculation(self, risk_calc, sample_data):
        """Test tracking error calculation."""
        strategy_returns, benchmark_returns = sample_data
        tracking_error = risk_calc.calculate_tracking_error(strategy_returns, benchmark_returns)
        
        assert isinstance(tracking_error, float)
        assert tracking_error >= 0
    
    def test_information_ratio_calculation(self, risk_calc, sample_data):
        """Test information ratio calculation."""
        strategy_returns, benchmark_returns = sample_data
        info_ratio = risk_calc.calculate_information_ratio(strategy_returns, benchmark_returns)
        
        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)
    
    def test_correlation_calculation(self, risk_calc, sample_data):
        """Test correlation calculation."""
        strategy_returns, benchmark_returns = sample_data
        correlation = risk_calc.calculate_correlation(strategy_returns, benchmark_returns)
        
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0
    
    def test_downside_deviation_calculation(self, risk_calc, sample_data):
        """Test downside deviation calculation."""
        strategy_returns, _ = sample_data
        downside_dev = risk_calc.calculate_downside_deviation(strategy_returns)
        
        assert isinstance(downside_dev, float)
        assert downside_dev >= 0
    
    def test_ulcer_index_calculation(self, risk_calc, sample_data):
        """Test Ulcer Index calculation."""
        strategy_returns, _ = sample_data
        cumulative_returns = (1 + strategy_returns).cumprod()
        ulcer_index = risk_calc.calculate_ulcer_index(cumulative_returns)
        
        assert isinstance(ulcer_index, float)
        assert ulcer_index >= 0
    
    def test_gain_to_pain_ratio_calculation(self, risk_calc, sample_data):
        """Test gain-to-pain ratio calculation."""
        strategy_returns, _ = sample_data
        gtp_ratio = risk_calc.calculate_gain_to_pain_ratio(strategy_returns)
        
        assert isinstance(gtp_ratio, float)
        assert gtp_ratio >= 0
    
    def test_all_risk_metrics_calculation(self, risk_calc, sample_data):
        """Test calculating all risk metrics."""
        strategy_returns, benchmark_returns = sample_data
        risk_metrics = risk_calc.calculate_all_risk_metrics(strategy_returns, benchmark_returns)
        
        expected_metrics = [
            'volatility', 'downside_deviation', 'ulcer_index',
            'gain_to_pain_ratio', 'beta', 'tracking_error',
            'information_ratio', 'correlation'
        ]
        
        for metric in expected_metrics:
            assert metric in risk_metrics
            assert isinstance(risk_metrics[metric], (int, float))
    
    def test_edge_case_zero_volatility(self, risk_calc):
        """Test edge case with zero volatility."""
        # Create zero volatility returns
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        zero_vol_returns = pd.Series([0.01] * len(dates), index=dates)
        
        risk_metrics = risk_calc.calculate_all_risk_metrics(zero_vol_returns)
        
        # Should handle zero volatility gracefully
        assert 'volatility' in risk_metrics
        # Account for floating point precision - should be very close to zero
        assert risk_metrics['volatility'] < 1e-10
    
    def test_edge_case_single_return(self, risk_calc):
        """Test edge case with single return value."""
        single_return = pd.Series([0.01], index=[datetime(2023, 1, 1)])
        
        risk_metrics = risk_calc.calculate_all_risk_metrics(single_return)
        
        # Should handle single value gracefully
        assert 'volatility' in risk_metrics
