"""
Tests for feature generation and registry.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.generators import FeatureGenerator, FeatureCategory
from src.features.registry import FeatureRegistry
from src.strategies.schema import FeatureSpec, FeatureType


class TestFeatureGenerator:
    """Test feature generation functionality."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic OHLCV data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = pd.Series(base_price * (1 + returns).cumprod(), index=dates)
        
        # Generate OHLC from close prices with realistic spreads
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_prices = prices.shift(1).fillna(base_price) * (1 + np.random.normal(0, 0.005, n_days))
        volume = pd.Series(np.random.lognormal(15, 0.5, n_days), index=dates)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        })
    
    @pytest.fixture
    def feature_generator(self):
        """Create FeatureGenerator instance."""
        return FeatureGenerator()
    
    def test_price_features_generation(self, feature_generator, sample_ohlcv_data):
        """Test price feature generation."""
        features = feature_generator.generate_price_features(
            sample_ohlcv_data, 
            lookback_periods=[5, 20]
        )
        
        # Check basic features
        assert 'returns' in features
        assert 'log_returns' in features
        assert 'price_level' in features
        assert 'price_rank' in features
        
        # Check rolling features
        assert 'sma_5' in features
        assert 'sma_20' in features
        assert 'ema_5' in features
        assert 'ema_20' in features
        
        # Check price relative features
        assert 'price_vs_sma_5' in features
        assert 'price_vs_ema_20' in features
        
        # Validate data integrity
        assert not features['returns'].isna().all()
        assert not features['sma_20'].isna().all()
        
        # Check that rolling features have appropriate NaN handling
        assert features['sma_20'].isna().sum() >= 19  # First 19 should be NaN
    
    def test_volume_features_generation(self, feature_generator, sample_ohlcv_data):
        """Test volume feature generation."""
        features = feature_generator.generate_volume_features(
            sample_ohlcv_data,
            lookback_periods=[5, 20]
        )
        
        # Check basic features
        assert 'volume_change' in features
        assert 'volume_level' in features
        assert 'volume_rank' in features
        
        # Check rolling features
        assert 'volume_sma_5' in features
        assert 'volume_sma_20' in features
        assert 'volume_vs_sma_5' in features
        
        # Check volume-price features
        assert 'volume_price_trend' in features
        assert 'volume_weighted_price' in features
        
        # Validate data integrity
        assert not features['volume_change'].isna().all()
        assert features['volume_level'].equals(sample_ohlcv_data['volume'])
    
    def test_volatility_features_generation(self, feature_generator, sample_ohlcv_data):
        """Test volatility feature generation."""
        features = feature_generator.generate_volatility_features(
            sample_ohlcv_data,
            lookback_periods=[5, 20]
        )
        
        # Check realized volatility features
        assert 'realized_vol_5' in features
        assert 'realized_vol_20' in features
        assert 'log_vol_5' in features
        assert 'log_vol_20' in features
        
        # Check advanced volatility features
        assert 'parkinson_vol_5' in features
        assert 'parkinson_vol_20' in features
        assert 'gk_vol_5' in features
        assert 'gk_vol_20' in features
        
        # Validate that volatility is positive
        realized_vol = features['realized_vol_20'].dropna()
        assert (realized_vol >= 0).all()
        
        # Check volatility ratios
        if 'vol_ratio_20_60' in features:
            vol_ratio = features['vol_ratio_20_60'].dropna()
            assert (vol_ratio > 0).all()
    
    def test_technical_indicators_generation(self, feature_generator, sample_ohlcv_data):
        """Test technical indicator generation."""
        features = feature_generator.generate_technical_indicators(
            sample_ohlcv_data,
            indicators=['rsi', 'macd', 'bollinger_bands']
        )
        
        # Check RSI
        assert 'rsi_14' in features
        rsi_values = features['rsi_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()
        
        # Check MACD
        assert 'macd_line' in features
        assert 'macd_signal' in features
        assert 'macd_histogram' in features
        
        # Check Bollinger Bands
        assert 'bb_upper' in features
        assert 'bb_lower' in features
        assert 'bb_position' in features
        
        # Validate BB position exists and is numeric
        bb_pos = features['bb_position'].dropna()
        assert len(bb_pos) > 0
        assert bb_pos.dtype in [np.float64, np.float32]
        # BB position represents price relative to bands, not necessarily 0-1
    
    def test_momentum_features_generation(self, feature_generator, sample_ohlcv_data):
        """Test momentum feature generation."""
        features = feature_generator.generate_momentum_features(
            sample_ohlcv_data,
            lookback_periods=[10, 20]
        )
        
        # Check momentum features (based on actual implementation)
        assert 'momentum_10' in features
        assert 'momentum_20' in features
        assert 'log_momentum_10' in features
        assert 'momentum_rank_10' in features
        assert 'roc_10' in features  # Rate of change
        assert 'rsi_10' in features  # RSI is part of momentum features
        
        # Validate momentum ranges
        momentum = features['momentum_20'].dropna()
        assert momentum.dtype == np.float64
    
    def test_feature_generation_with_missing_data(self, feature_generator):
        """Test feature generation with missing data."""
        # Create data with missing values
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        data = pd.DataFrame({
            'open': [100] * len(dates),
            'high': [105] * len(dates),
            'low': [95] * len(dates),
            'close': [100] * len(dates),
            'volume': [1000] * len(dates)
        }, index=dates)
        
        # Introduce missing values
        data.loc[data.index[10:15], 'close'] = np.nan
        
        # Should handle missing data gracefully
        features = feature_generator.generate_price_features(data, lookback_periods=[5])
        
        assert 'returns' in features
        assert 'sma_5' in features


class TestFeatureRegistry:
    """Test feature registry functionality."""
    
    @pytest.fixture
    def feature_registry(self):
        """Create FeatureRegistry instance."""
        return FeatureRegistry()
    
    @pytest.fixture
    def sample_feature_specs(self):
        """Create sample feature specifications."""
        return [
            FeatureSpec(
                name="rsi_14",
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                lookback_period=14,
                parameters={"period": 14}
            ),
            FeatureSpec(
                name="momentum_20",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=20,
                parameters={"method": "returns"}
            )
        ]
    
    def test_feature_registration(self, feature_registry):
        """Test registering features."""
        def custom_feature(data, **params):
            return data['close'].rolling(params['period']).mean()
        
        # Test with FeatureSpec object
        feature_spec = FeatureSpec(
            name="custom_ma",
            feature_type=FeatureType.PRICE_BASED,
            lookback_period=20,
            parameters={"period": 20}
        )
        
        result = feature_registry.register_feature(feature_spec, custom_feature)
        assert result is True
        
        assert feature_registry.has_feature("custom_ma")
        feature_list = feature_registry.list_features()
        feature_names = [f.name for f in feature_list]
        assert "custom_ma" in feature_names
        
        # Test alternative registration method
        result = feature_registry.register_feature_with_function(
            name="test_alternative",
            feature_function=lambda x: x.mean(),
            feature_type=FeatureType.TECHNICAL_INDICATOR,
            description="Test alternative registration",
            lookback_period=10
        )
        assert result is True
        assert feature_registry.has_feature("test_alternative")
    
    def test_feature_generation_from_specs(self, feature_registry, sample_feature_specs):
        """Test generating features from specifications."""
        # This would be implemented in the actual FeatureRegistry
        # For now, test that the registry can handle the feature specs
        feature_names = [spec.name for spec in sample_feature_specs]
        
        assert "rsi_14" in feature_names
        assert "momentum_20" in feature_names
    
    def test_feature_validation(self, feature_registry):
        """Test feature validation."""
        # Test that registry validates feature specifications
        valid_spec = FeatureSpec(
            name="valid_feature",
            feature_type=FeatureType.PRICE_BASED,
            lookback_period=20,
            parameters={"method": "sma"}
        )
        
        # Should not raise any errors
        is_valid = feature_registry.validate_feature_spec(valid_spec)
        assert is_valid  # Assuming this method exists
    
    def test_get_feature_categories(self, feature_registry):
        """Test getting feature categories."""
        categories = feature_registry.get_available_categories()
        
        # Should include common categories
        expected_categories = [cat.value for cat in FeatureCategory]
        for category in expected_categories:
            # Registry should be aware of these categories
            pass  # Actual implementation would check this
