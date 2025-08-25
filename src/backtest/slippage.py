"""
Slippage models for realistic backtesting.

This module provides various slippage models to simulate
real-world trading conditions in backtests.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SlippageModel:
    """
    Base slippage model for backtesting.
    
    Slippage represents the difference between the expected price
    and the actual execution price due to market impact.
    """
    
    def __init__(self, slippage_bps: float = 5.0):
        """
        Initialize slippage model.
        
        Args:
            slippage_bps: Slippage in basis points (default: 5 bps)
        """
        self.slippage_bps = slippage_bps
        self.slippage_rate = slippage_bps / 10000  # Convert to decimal
        
        logger.info(f"Slippage model initialized with {slippage_bps} basis points")
    
    def calculate_slippage(self, price: float, quantity: float) -> float:
        """
        Calculate slippage cost for a trade.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            Slippage cost in currency units
        """
        trade_value = price * quantity
        slippage_cost = trade_value * self.slippage_rate
        
        return slippage_cost


class VolumeWeightedSlippageModel(SlippageModel):
    """
    Volume-weighted slippage model.
    
    This model adjusts slippage based on the trade size relative
    to average daily volume.
    """
    
    def __init__(self, slippage_bps: float = 5.0, volume_impact_factor: float = 0.1):
        """
        Initialize volume-weighted slippage model.
        
        Args:
            slippage_bps: Base slippage in basis points
            volume_impact_factor: Impact factor for volume-based adjustment
        """
        super().__init__(slippage_bps)
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(
        self, 
        price: float, 
        quantity: float, 
        avg_daily_volume: Optional[float] = None
    ) -> float:
        """
        Calculate volume-weighted slippage.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            avg_daily_volume: Average daily volume (optional)
            
        Returns:
            Adjusted slippage cost
        """
        base_slippage = super().calculate_slippage(price, quantity)
        
        if avg_daily_volume and avg_daily_volume > 0:
            # Adjust slippage based on trade size relative to volume
            volume_ratio = quantity / avg_daily_volume
            volume_adjustment = 1 + (volume_ratio * self.volume_impact_factor)
            adjusted_slippage = base_slippage * volume_adjustment
            
            return adjusted_slippage
        
        return base_slippage


class VolatilityWeightedSlippageModel(SlippageModel):
    """
    Volatility-weighted slippage model.
    
    This model adjusts slippage based on market volatility,
    with higher volatility leading to higher slippage.
    """
    
    def __init__(self, slippage_bps: float = 5.0, volatility_impact_factor: float = 2.0):
        """
        Initialize volatility-weighted slippage model.
        
        Args:
            slippage_bps: Base slippage in basis points
            volatility_impact_factor: Impact factor for volatility adjustment
        """
        super().__init__(slippage_bps)
        self.volatility_impact_factor = volatility_impact_factor
    
    def calculate_slippage(
        self, 
        price: float, 
        quantity: float, 
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate volatility-weighted slippage.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            volatility: Market volatility (optional)
            
        Returns:
            Adjusted slippage cost
        """
        base_slippage = super().calculate_slippage(price, quantity)
        
        if volatility and volatility > 0:
            # Adjust slippage based on volatility
            # Higher volatility = higher slippage
            volatility_adjustment = 1 + (volatility * self.volatility_impact_factor)
            adjusted_slippage = base_slippage * volatility_adjustment
            
            return adjusted_slippage
        
        return base_slippage


class AdaptiveSlippageModel(SlippageModel):
    """
    Adaptive slippage model that combines multiple factors.
    
    This model adjusts slippage based on:
    - Trade size relative to volume
    - Market volatility
    - Time of day
    - Market conditions
    """
    
    def __init__(
        self,
        slippage_bps: float = 5.0,
        volume_impact_factor: float = 0.1,
        volatility_impact_factor: float = 2.0,
        time_impact_factor: float = 0.5
    ):
        """
        Initialize adaptive slippage model.
        
        Args:
            slippage_bps: Base slippage in basis points
            volume_impact_factor: Impact factor for volume-based adjustment
            volatility_impact_factor: Impact factor for volatility adjustment
            time_impact_factor: Impact factor for time-based adjustment
        """
        super().__init__(slippage_bps)
        self.volume_impact_factor = volume_impact_factor
        self.volatility_impact_factor = volatility_impact_factor
        self.time_impact_factor = time_impact_factor
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        avg_daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,
        time_of_day: Optional[float] = None,
        market_conditions: Optional[str] = None
    ) -> float:
        """
        Calculate adaptive slippage based on multiple factors.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            volatility: Market volatility
            time_of_day: Time of day (0-1, where 0.5 is market open)
            market_conditions: Market condition indicator
            
        Returns:
            Comprehensive slippage cost
        """
        base_slippage = super().calculate_slippage(price, quantity)
        
        # Volume adjustment
        volume_adjustment = 1.0
        if avg_daily_volume and avg_daily_volume > 0:
            volume_ratio = quantity / avg_daily_volume
            volume_adjustment = 1 + (volume_ratio * self.volume_impact_factor)
        
        # Volatility adjustment
        volatility_adjustment = 1.0
        if volatility and volatility > 0:
            volatility_adjustment = 1 + (volatility * self.volatility_impact_factor)
        
        # Time adjustment (higher slippage at market open/close)
        time_adjustment = 1.0
        if time_of_day is not None:
            # Higher impact at market open (0.0) and close (1.0)
            time_distance_from_center = abs(time_of_day - 0.5) * 2
            time_adjustment = 1 + (time_distance_from_center * self.time_impact_factor)
        
        # Market conditions adjustment
        market_adjustment = 1.0
        if market_conditions:
            if market_conditions.lower() in ['high_volatility', 'crisis', 'panic']:
                market_adjustment = 1.5
            elif market_conditions.lower() in ['low_volatility', 'calm', 'stable']:
                market_adjustment = 0.8
        
        # Combine all adjustments
        total_adjustment = volume_adjustment * volatility_adjustment * time_adjustment * market_adjustment
        adjusted_slippage = base_slippage * total_adjustment
        
        return adjusted_slippage


class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model with constant rate.
    
    This is the simplest slippage model that applies
    a fixed percentage to all trades.
    """
    
    def __init__(self, slippage_bps: float = 5.0):
        """
        Initialize fixed slippage model.
        
        Args:
            slippage_bps: Fixed slippage in basis points
        """
        super().__init__(slippage_bps)
    
    def calculate_slippage(self, price: float, quantity: float) -> float:
        """
        Calculate fixed slippage cost.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            Fixed slippage cost
        """
        return super().calculate_slippage(price, quantity)


def create_slippage_model(model_type: str = "fixed", **kwargs) -> SlippageModel:
    """
    Factory function to create slippage models.
    
    Args:
        model_type: Type of slippage model
        **kwargs: Additional parameters for the model
        
    Returns:
        Configured slippage model instance
    """
    models = {
        "fixed": FixedSlippageModel,
        "volume_weighted": VolumeWeightedSlippageModel,
        "volatility_weighted": VolatilityWeightedSlippageModel,
        "adaptive": AdaptiveSlippageModel
    }
    
    if model_type not in models:
        logger.warning(f"Unknown slippage model type: {model_type}. Using fixed model.")
        model_type = "fixed"
    
    model_class = models[model_type]
    return model_class(**kwargs)
