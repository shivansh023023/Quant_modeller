"""
Feature generators for common financial features.

This module provides functions to generate various types of features
commonly used in quantitative trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from enum import Enum

from ..strategies.schema import FeatureSpec, FeatureType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Categories of financial features."""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TECHNICAL_INDICATOR = "technical_indicator"
    FUNDAMENTAL = "fundamental"
    CROSS_SECTIONAL = "cross_sectional"
    REGIME = "regime"


class FeatureGenerator:
    """
    Generate common financial features for trading strategies.
    
    This class provides methods to create various types of features
    including price-based, volume-based, volatility-based, and
    cross-sectional features.
    """
    
    def __init__(self):
        """Initialize the feature generator."""
        self.feature_cache = {}
    
    def generate_price_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate price-based features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60, 252]
        
        features = {}
        
        # Price returns
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price levels
        features['price_level'] = data['close']
        features['price_rank'] = data['close'].rank(pct=True)
        
        # Price changes
        features['price_change'] = data['close'] - data['close'].shift(1)
        features['price_change_pct'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Rolling price statistics
        for period in lookback_periods:
            if period <= len(data):
                # Moving averages
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
                features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                
                # Price relative to moving averages
                features[f'price_vs_sma_{period}'] = data['close'] / features[f'sma_{period}'] - 1
                features[f'price_vs_ema_{period}'] = data['close'] / features[f'ema_{period}'] - 1
                
                # Rolling percentiles
                features[f'price_percentile_{period}'] = data['close'].rolling(period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                
                # Rolling min/max
                features[f'price_vs_min_{period}'] = data['close'] / data['close'].rolling(period).min() - 1
                features[f'price_vs_max_{period}'] = data['close'] / data['close'].rolling(period).max() - 1
        
        return features
    
    def generate_volume_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate volume-based features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60]
        
        features = {}
        
        # Volume returns
        features['volume_change'] = data['volume'].pct_change()
        features['volume_change_abs'] = (data['volume'] - data['volume'].shift(1)).abs()
        
        # Volume levels
        features['volume_level'] = data['volume']
        features['volume_rank'] = data['volume'].rank(pct=True)
        
        # Rolling volume statistics
        for period in lookback_periods:
            if period <= len(data):
                # Moving averages
                features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ema_{period}'] = data['volume'].ewm(span=period).mean()
                
                # Volume relative to moving averages
                features[f'volume_vs_sma_{period}'] = data['volume'] / features[f'volume_sma_{period}'] - 1
                features[f'volume_vs_ema_{period}'] = data['volume'] / features[f'volume_ema_{period}'] - 1
                
                # Volume percentiles
                features[f'volume_percentile_{period}'] = data['volume'].rolling(period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
        
        # Volume-price relationship
        features['volume_price_trend'] = (data['volume'] * data['close']).pct_change()
        features['volume_weighted_price'] = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        return features
    
    def generate_volatility_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate volatility-based features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60]
        
        features = {}
        
        # Returns for volatility calculation
        returns = data['close'].pct_change()
        log_returns = np.log(data['close'] / data['close'].shift(1))
        
        # Realized volatility
        for period in lookback_periods:
            if period <= len(data):
                # Standard deviation
                features[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
                
                # Log returns volatility
                features[f'log_vol_{period}'] = log_returns.rolling(period).std() * np.sqrt(252)
                
                # Parkinson volatility (high-low based)
                hl_ratio = np.log(data['high'] / data['low'])
                features[f'parkinson_vol_{period}'] = np.sqrt(
                    (hl_ratio ** 2).rolling(period).mean() / (4 * np.log(2)) * 252
                )
                
                # Garman-Klass volatility
                c_o = np.log(data['close'] / data['open'])
                h_l = np.log(data['high'] / data['low'])
                o_l = np.log(data['open'] / data['low'])
                features[f'gk_vol_{period}'] = np.sqrt(
                    (0.5 * h_l ** 2 - (2 * np.log(2) - 1) * c_o ** 2).rolling(period).mean() * 252
                )
        
        # Volatility ratios
        if 20 in lookback_periods and 60 in lookback_periods:
            features['vol_ratio_20_60'] = features['realized_vol_20'] / features['realized_vol_60']
        
        # Volatility of volatility
        if 20 in lookback_periods:
            features['vol_of_vol_20'] = features['realized_vol_20'].rolling(20).std()
        
        return features
    
    def generate_technical_indicators(
        self, 
        data: pd.DataFrame, 
        indicators: List[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicators to generate
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if indicators is None:
            indicators = ['rsi', 'macd', 'bollinger_bands']
        
        features = {}
        
        if 'rsi' in indicators:
            # RSI (14-period)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
        
        if 'macd' in indicators:
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            features['macd_line'] = ema_12 - ema_26
            features['macd_signal'] = features['macd_line'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd_line'] - features['macd_signal']
        
        if 'bollinger_bands' in indicators:
            # Bollinger Bands (20-period)
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (2 * std_20)
            features['bb_lower'] = sma_20 - (2 * std_20)
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (data['close'] - features['bb_lower']) / features['bb_width']
        
        return features
    
    def generate_momentum_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate momentum-based features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60, 252]
        
        features = {}
        
        # Price momentum
        for period in lookback_periods:
            if period <= len(data):
                # Simple momentum
                features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                
                # Log momentum
                features[f'log_momentum_{period}'] = np.log(data['close'] / data['close'].shift(period))
                
                # Momentum rank
                features[f'momentum_rank_{period}'] = features[f'momentum_{period}'].rolling(period).rank(pct=True)
        
        # Rate of change
        for period in lookback_periods:
            if period <= len(data):
                features[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)
        
        # Relative strength index
        for period in lookback_periods:
            if period <= len(data):
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 12 in lookback_periods and 26 in lookback_periods:
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        return features
    
    def generate_mean_reversion_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate mean reversion features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60]
        
        features = {}
        
        # Bollinger Bands
        for period in lookback_periods:
            if period <= len(data):
                sma = data['close'].rolling(period).mean()
                std = data['close'].rolling(period).std()
                
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
                features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
        
        # Price deviation from trend
        for period in lookback_periods:
            if period <= len(data):
                sma = data['close'].rolling(period).mean()
                features[f'price_deviation_{period}'] = (data['close'] - sma) / sma
                
                # Z-score
                features[f'price_zscore_{period}'] = (data['close'] - sma) / data['close'].rolling(period).std()
        
        # Stochastic oscillator
        for period in lookback_periods:
            if period <= len(data):
                lowest_low = data['low'].rolling(period).min()
                highest_high = data['high'].rolling(period).max()
                features[f'stoch_k_{period}'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        return features
    
    def generate_technical_features(
        self, 
        data: pd.DataFrame, 
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate technical indicator features.
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [14, 20, 50]
        
        features = {}
        
        # Average True Range (ATR)
        for period in lookback_periods:
            if period <= len(data):
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift(1))
                low_close = np.abs(data['low'] - data['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features[f'atr_{period}'] = true_range.rolling(period).mean()
        
        # Williams %R
        for period in lookback_periods:
            if period <= len(data):
                highest_high = data['high'].rolling(period).max()
                lowest_low = data['low'].rolling(period).min()
                features[f'williams_r_{period}'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        for period in lookback_periods:
            if period <= len(data):
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                sma_tp = typical_price.rolling(period).mean()
                mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                features[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        for period in lookback_periods:
            if period <= len(data):
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                money_flow = typical_price * data['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
                
                money_ratio = positive_flow / negative_flow
                features[f'mfi_{period}'] = 100 - (100 / (1 + money_ratio))
        
        return features
    
    def generate_cross_sectional_features(
        self, 
        data: pd.DataFrame, 
        universe: List[str],
        lookback_periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate cross-sectional features.
        
        Args:
            data: DataFrame with multi-index (date, ticker) or panel data
            universe: List of ticker symbols
            lookback_periods: List of lookback periods to use
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        if lookback_periods is None:
            lookback_periods = [5, 20, 60]
        
        features = {}
        
        # Ensure data is properly formatted for cross-sectional analysis
        if not isinstance(data.index, pd.MultiIndex):
            logger.warning("Data should have MultiIndex (date, ticker) for cross-sectional features")
            return features
        
        # Cross-sectional rank features
        for period in lookback_periods:
            if period <= len(data):
                # Price rank across universe
                features[f'price_rank_{period}'] = data.groupby(level=0)['close'].rank(pct=True)
                
                # Volume rank across universe
                features[f'volume_rank_{period}'] = data.groupby(level=0)['volume'].rank(pct=True)
                
                # Returns rank across universe
                returns = data.groupby(level=1)['close'].pct_change()
                features[f'returns_rank_{period}'] = returns.groupby(level=0).rank(pct=True)
        
        # Cross-sectional z-scores
        for period in lookback_periods:
            if period <= len(data):
                # Price z-score
                price_mean = data.groupby(level=0)['close'].rolling(period).mean().reset_index(0, drop=True)
                price_std = data.groupby(level=0)['close'].rolling(period).std().reset_index(0, drop=True)
                features[f'price_zscore_{period}'] = (data['close'] - price_mean) / price_std
                
                # Volume z-score
                volume_mean = data.groupby(level=0)['volume'].rolling(period).mean().reset_index(0, drop=True)
                volume_std = data.groupby(level=0)['volume'].rolling(period).std().reset_index(0, drop=True)
                features[f'volume_zscore_{period}'] = (data['volume'] - volume_mean) / volume_std
        
        # Cross-sectional momentum
        for period in lookback_periods:
            if period <= len(data):
                momentum = data.groupby(level=1)['close'].pct_change(period)
                features[f'cross_momentum_{period}'] = momentum.groupby(level=0).rank(pct=True)
        
        return features
    
    def generate_regime_features(
        self, 
        data: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate market regime features.
        
        Args:
            data: DataFrame with OHLCV data
            market_data: Market-wide data (optional)
            
        Returns:
            Dictionary of feature names to pandas Series
        """
        features = {}
        
        # Volatility regime
        returns = data['close'].pct_change()
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_60 = returns.rolling(60).std() * np.sqrt(252)
        
        features['vol_regime'] = np.where(vol_20 > vol_60, 'high_vol', 'low_vol')
        features['vol_regime_numeric'] = np.where(vol_20 > vol_60, 1, 0)
        
        # Trend regime
        sma_20 = data['close'].rolling(20).mean()
        sma_60 = data['close'].rolling(60).mean()
        
        features['trend_regime'] = np.where(sma_20 > sma_60, 'uptrend', 'downtrend')
        features['trend_regime_numeric'] = np.where(sma_20 > sma_60, 1, 0)
        
        # Market stress (if market data available)
        if market_data is not None:
            if 'vix' in market_data.columns:
                vix = market_data['vix']
                features['stress_regime'] = np.where(vix > vix.rolling(252).quantile(0.8), 'high_stress', 'low_stress')
                features['stress_regime_numeric'] = np.where(vix > vix.rolling(252).quantile(0.8), 1, 0)
        
        return features
    
    def generate_all_features(
        self, 
        data: pd.DataFrame,
        universe: Optional[List[str]] = None,
        lookback_periods: List[int] = None,
        include_regime: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Generate all available features.
        
        Args:
            data: DataFrame with OHLCV data
            universe: List of ticker symbols (for cross-sectional features)
            lookback_periods: List of lookback periods to use
            include_regime: Whether to include regime features
            
        Returns:
            Dictionary of all feature names to pandas Series
        """
        all_features = {}
        
        # Generate basic features
        all_features.update(self.generate_price_features(data, lookback_periods))
        all_features.update(self.generate_volume_features(data, lookback_periods))
        all_features.update(self.generate_volatility_features(data, lookback_periods))
        all_features.update(self.generate_momentum_features(data, lookback_periods))
        all_features.update(self.generate_mean_reversion_features(data, lookback_periods))
        all_features.update(self.generate_technical_features(data, lookback_periods))
        
        # Generate cross-sectional features if universe provided
        if universe and len(data) > 0:
            all_features.update(self.generate_cross_sectional_features(data, universe, lookback_periods))
        
        # Generate regime features
        if include_regime:
            all_features.update(self.generate_regime_features(data))
        
        # Clean up features (remove NaN values)
        for name, feature in all_features.items():
            if isinstance(feature, pd.Series):
                all_features[name] = feature.fillna(method='ffill').fillna(0)
        
        logger.info(f"Generated {len(all_features)} features")
        return all_features
    
    def create_feature_specs(
        self, 
        feature_names: List[str],
        feature_type: FeatureType = FeatureType.TECHNICAL_INDICATOR
    ) -> List[FeatureSpec]:
        """
        Create FeatureSpec objects from feature names.
        
        Args:
            feature_names: List of feature names
            feature_type: Type of features to create
            
        Returns:
            List of FeatureSpec objects
        """
        specs = []
        
        for name in feature_names:
            # Extract lookback period from name if present
            lookback_period = 20  # Default
            for part in name.split('_'):
                if part.isdigit():
                    lookback_period = int(part)
                    break
            
            # Create description based on feature name
            description = f"Generated {name} feature with {lookback_period}-day lookback"
            
            spec = FeatureSpec(
                name=name,
                description=description,
                feature_type=feature_type,
                lookback_period=lookback_period,
                data_source="ohlcv"
            )
            
            specs.append(spec)
        
        return specs
    
    def get_feature_categories(self) -> Dict[FeatureCategory, List[str]]:
        """
        Get available feature categories and their descriptions.
        
        Returns:
            Dictionary mapping feature categories to descriptions
        """
        return {
            FeatureCategory.PRICE: "Price-based features including returns, moving averages, and price levels",
            FeatureCategory.VOLUME: "Volume-based features including volume changes and volume-price relationships",
            FeatureCategory.VOLATILITY: "Volatility measures including realized volatility and GARCH estimates",
            FeatureCategory.MOMENTUM: "Momentum indicators including RSI, MACD, and price momentum",
            FeatureCategory.MEAN_REVERSION: "Mean reversion indicators including Bollinger Bands and z-scores",
            FeatureCategory.TECHNICAL_INDICATOR: "Technical indicators including ATR, Williams %R, and CCI",
            FeatureCategory.FUNDAMENTAL: "Fundamental data features (not implemented yet)",
            FeatureCategory.CROSS_SECTIONAL: "Cross-sectional features across universe of securities",
            FeatureCategory.REGIME: "Market regime classification features"
        }
