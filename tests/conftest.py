"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def sample_market_data():
    """Create realistic sample market data for testing."""
    np.random.seed(42)
    
    # Create 2 years of daily data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic price movements
    base_price = 100.0
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # ~0.13% daily return, 1.5% volatility
    prices = base_price * (1 + daily_returns).cumprod()
    
    # Generate OHLC from close prices
    daily_volatility = np.random.uniform(0.005, 0.025, n_days)
    high = prices * (1 + daily_volatility)
    low = prices * (1 - daily_volatility)
    open_prices = prices.shift(1).fillna(base_price) * (1 + np.random.normal(0, 0.003, n_days))
    
    # Generate volume
    base_volume = 1_000_000
    volume_volatility = np.random.normal(0, 0.3, n_days)
    volume = base_volume * np.exp(volume_volatility)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'returns': daily_returns
    }, index=dates)


@pytest.fixture(scope="session")
def sample_multi_asset_data():
    """Create sample data for multiple assets."""
    np.random.seed(42)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    data = {}
    for ticker in tickers:
        # Create correlated but distinct price series
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100.0 * (1 + returns).cumprod()
        
        data[ticker] = pd.DataFrame({
            'open': prices.shift(1).fillna(100.0),
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, len(dates)),
            'returns': returns
        }, index=dates)
    
    return data


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        'alpha_vantage': {
            'api_key': 'test_alpha_key',
            'rate_limit': 5,
            'enabled': True
        },
        'gemini': {
            'api_key': 'test_gemini_key',
            'model': 'gemini-pro',
            'enabled': True
        },
        'yfinance': {
            'enabled': True,
            'rate_limit': 2
        }
    }


@pytest.fixture
def sample_strategy_spec():
    """Create a sample strategy specification for testing."""
    from src.strategies.schema import (
        StrategySpec, FeatureSpec, EntryExitRule, PositionSizing,
        CrossValidationConfig, FeatureType, TargetType, ValidationType
    )
    
    return StrategySpec(
        name="Test Strategy",
        description="A test strategy for unit testing",
        universe=["AAPL", "GOOGL", "MSFT"],
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        features=[
            FeatureSpec(
                name="momentum_20",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=20,
                parameters={"method": "returns"}
            ),
            FeatureSpec(
                name="volume_ratio_20",
                feature_type=FeatureType.VOLUME_BASED,
                lookback_period=20,
                parameters={"method": "volume_ratio"}
            )
        ],
        target=TargetType.NEXT_DAY_RETURN,
        target_lookback=1,
        entry_rules=EntryExitRule(
            condition="momentum_20 > 0.02 and volume_ratio_20 > 1.2",
            lookback_period=20
        ),
        exit_rules=EntryExitRule(
            condition="momentum_20 < -0.01",
            lookback_period=10
        ),
        holding_period=10,
        position_sizing=PositionSizing(
            method="volatility_target",
            volatility_target=0.15,
            max_position_size=0.2
        ),
        max_positions=5,
        cv_config=CrossValidationConfig(
            validation_type=ValidationType.WALK_FORWARD,
            n_splits=5,
            train_size=0.7,
            purge_period=10,
            embargo_period=5
        ),
        tags=["test", "momentum"]
    )


@pytest.fixture
def sample_returns_series():
    """Create sample returns series for performance testing."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # Generate returns with some autocorrelation and volatility clustering
    returns = np.random.normal(0.0005, 0.015, len(dates))
    
    # Add some volatility clustering
    volatility = np.random.uniform(0.01, 0.03, len(dates))
    returns = returns * volatility
    
    return pd.Series(returns, index=dates, name='strategy_returns')


# Test data validation utilities
def assert_valid_dataframe(df, required_columns=None, min_rows=1):
    """Assert that DataFrame meets basic requirements."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= min_rows
    assert not df.empty
    
    if required_columns:
        for col in required_columns:
            assert col in df.columns


def assert_valid_series(series, min_length=1, check_finite=True):
    """Assert that Series meets basic requirements."""
    assert isinstance(series, pd.Series)
    assert len(series) >= min_length
    
    if check_finite:
        finite_values = series.dropna()
        if len(finite_values) > 0:
            assert np.isfinite(finite_values).all()


def assert_metric_value(value, expected_range=None, allow_inf=False):
    """Assert that a metric value is reasonable."""
    if not allow_inf:
        assert np.isfinite(value), f"Metric value should be finite, got {value}"
    
    if expected_range:
        min_val, max_val = expected_range
        assert min_val <= value <= max_val, f"Metric value {value} outside expected range {expected_range}"


# Performance test utilities
def benchmark_function(func, *args, max_time_seconds=10, **kwargs):
    """Benchmark a function and ensure it completes within time limit."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < max_time_seconds, f"Function took {execution_time:.2f}s, expected < {max_time_seconds}s"
    
    return result, execution_time
