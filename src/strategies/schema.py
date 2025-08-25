"""
Strategy specification schema using Pydantic.

This module defines the core StrategySpec class that represents
a complete quantitative trading strategy specification.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator
import pandas as pd


class TargetType(str, Enum):
    """Types of prediction targets."""
    NEXT_DAY_RETURN = "next_day_return"
    NEXT_DAY_DIRECTION = "next_day_direction"
    NEXT_WEEK_RETURN = "next_week_return"
    NEXT_MONTH_RETURN = "next_month_return"
    CUSTOM = "custom"


class FeatureType(str, Enum):
    """Types of features that can be engineered."""
    PRICE_BASED = "price_based"
    VOLUME_BASED = "volume_based"
    TECHNICAL_INDICATOR = "technical_indicator"
    FUNDAMENTAL = "fundamental"
    MARKET_NEUTRAL = "market_neutral"
    CUSTOM = "custom"


class ValidationType(str, Enum):
    """Types of cross-validation strategies."""
    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    TIME_SERIES_SPLIT = "time_series_split"


class FeatureSpec(BaseModel):
    """Specification for a single feature."""
    name: str = Field(..., description="Feature name")
    feature_type: FeatureType = Field(..., description="Type of feature")
    lookback_period: int = Field(..., gt=0, description="Lookback period in days")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Feature parameters")
    
    @validator('lookback_period')
    def validate_lookback(cls, v):
        """Ensure lookback period is reasonable."""
        if v > 252:  # More than 1 trading year
            raise ValueError("Lookback period cannot exceed 252 days (1 trading year)")
        return v


class EntryExitRule(BaseModel):
    """Entry or exit rule specification."""
    condition: str = Field(..., description="Rule condition as string expression")
    lookback_period: int = Field(..., gt=0, description="Lookback period for rule evaluation")
    
    @validator('lookback_period')
    def validate_lookback(cls, v):
        """Ensure lookback period is reasonable."""
        if v > 252:
            raise ValueError("Rule lookback period cannot exceed 252 days")
        return v


class PositionSizing(BaseModel):
    """Position sizing specification."""
    method: str = Field(..., description="Position sizing method")
    volatility_target: Optional[float] = Field(None, ge=0.01, le=0.5, description="Volatility target (0.01-0.5)")
    risk_per_trade: Optional[float] = Field(None, ge=0.001, le=0.1, description="Risk per trade (0.001-0.1)")
    max_position_size: Optional[float] = Field(None, ge=0.01, le=1.0, description="Maximum position size (0.01-1.0)")
    
    @validator('method')
    def validate_method(cls, v):
        """Validate position sizing method."""
        valid_methods = ['volatility_target', 'risk_per_trade', 'fixed_size', 'kelly_criterion']
        if v not in valid_methods:
            raise ValueError(f"Position sizing method must be one of {valid_methods}")
        return v


class CrossValidationConfig(BaseModel):
    """Cross-validation configuration."""
    validation_type: ValidationType = Field(..., description="Type of validation")
    n_splits: int = Field(5, ge=2, le=20, description="Number of CV splits")
    train_size: float = Field(0.7, gt=0.5, lt=0.9, description="Training set size ratio")
    purge_period: int = Field(10, ge=0, le=50, description="Purge period for purged CV")
    embargo_period: int = Field(5, ge=0, le=20, description="Embargo period for purged CV")


class StrategySpec(BaseModel):
    """Complete strategy specification."""
    
    # Basic identification
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    version: str = Field("1.0.0", description="Strategy version")
    
    # Universe and time period
    universe: List[str] = Field(..., min_items=1, description="List of ticker symbols")
    start_date: datetime = Field(..., description="Strategy start date")
    end_date: datetime = Field(..., description="Strategy end date")
    
    # Features and target
    features: List[FeatureSpec] = Field(..., min_items=1, description="List of features")
    target: TargetType = Field(..., description="Prediction target")
    target_lookback: int = Field(1, ge=1, le=5, description="Target lookback period")
    
    # Entry/exit rules
    entry_rules: EntryExitRule = Field(..., description="Entry rule specification")
    exit_rules: EntryExitRule = Field(..., description="Exit rule specification")
    holding_period: int = Field(..., ge=1, le=252, description="Maximum holding period in days")
    
    # Risk management
    position_sizing: PositionSizing = Field(..., description="Position sizing specification")
    max_positions: int = Field(10, ge=1, le=100, description="Maximum number of concurrent positions")
    stop_loss: Optional[float] = Field(None, ge=0.01, le=0.5, description="Stop loss threshold")
    take_profit: Optional[float] = Field(None, ge=0.01, le=1.0, description="Take profit threshold")
    
    # Validation
    cv_config: CrossValidationConfig = Field(..., description="Cross-validation configuration")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Strategy tags")
    author: str = Field("Unknown", description="Strategy author")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
    
    @model_validator(mode='after')
    def validate_dates(self):
        """Validate date ranges and ensure no data leakage."""
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("Start date must be before end date")
            
            # Ensure minimum data period
            min_period = timedelta(days=252)  # At least 1 trading year
            if (self.end_date - self.start_date) < min_period:
                raise ValueError("Strategy period must be at least 1 trading year")
        
        return self
    
    @model_validator(mode='after')
    def validate_feature_lookbacks(self):
        """Ensure feature lookbacks don't exceed strategy period."""
        if self.features and self.start_date and self.end_date:
            strategy_days = (self.end_date - self.start_date).days
            
            for feature in self.features:
                if feature.lookback_period >= strategy_days:
                    raise ValueError(
                        f"Feature {feature.name} lookback period ({feature.lookback_period}) "
                        f"must be less than strategy period ({strategy_days} days)"
                    )
        
        return self
    
    @model_validator(mode='after')
    def validate_rules(self):
        """Ensure entry/exit rules don't cause data leakage."""
        if self.entry_rules and self.exit_rules and self.features:
            # Check that rule lookbacks don't exceed feature lookbacks
            max_feature_lookback = max(f.lookback_period for f in self.features)
            
            if self.entry_rules.lookback_period > max_feature_lookback:
                raise ValueError(
                    f"Entry rule lookback ({self.entry_rules.lookback_period}) "
                    f"cannot exceed max feature lookback ({max_feature_lookback})"
                )
            
            if self.exit_rules.lookback_period > max_feature_lookback:
                raise ValueError(
                    f"Exit rule lookback ({self.exit_rules.lookback_period}) "
                    f"cannot exceed max feature lookback ({max_feature_lookback})"
                )
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy spec to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert strategy spec to JSON string."""
        return self.json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "StrategySpec":
        """Create strategy spec from JSON string."""
        return cls.parse_raw(json_str)
    
    def get_max_lookback(self) -> int:
        """Get the maximum lookback period across all features and rules."""
        lookbacks = []
        
        # Feature lookbacks
        lookbacks.extend([f.lookback_period for f in self.features])
        
        # Rule lookbacks
        if self.entry_rules:
            lookbacks.append(self.entry_rules.lookback_period)
        if self.exit_rules:
            lookbacks.append(self.exit_rules.lookback_period)
        
        return max(lookbacks) if lookbacks else 0
    
    def validate_no_data_leakage(self) -> bool:
        """Validate that the strategy has no data leakage."""
        try:
            # This will raise validation errors if there's data leakage
            self.validate_feature_lookbacks(self.dict())
            self.validate_rules(self.dict())
            return True
        except ValueError:
            return False


class StrategyBuilder:
    """Builder class for creating StrategySpec instances."""
    
    def __init__(self):
        self._spec = {}
    
    def with_name(self, name: str) -> "StrategyBuilder":
        """Set strategy name."""
        self._spec['name'] = name
        return self
    
    def with_description(self, description: str) -> "StrategyBuilder":
        """Set strategy description."""
        self._spec['description'] = description
        return self
    
    def with_universe(self, universe: List[str]) -> "StrategyBuilder":
        """Set strategy universe."""
        self._spec['universe'] = universe
        return self
    
    def with_dates(self, start_date: datetime, end_date: datetime) -> "StrategyBuilder":
        """Set strategy date range."""
        self._spec['start_date'] = start_date
        self._spec['end_date'] = end_date
        return self
    
    def with_features(self, features: List[FeatureSpec]) -> "StrategyBuilder":
        """Set strategy features."""
        self._spec['features'] = features
        return self
    
    def with_target(self, target: TargetType, lookback: int = 1) -> "StrategyBuilder":
        """Set prediction target."""
        self._spec['target'] = target
        self._spec['target_lookback'] = lookback
        return self
    
    def with_rules(self, entry_rules: EntryExitRule, exit_rules: EntryExitRule) -> "StrategyBuilder":
        """Set entry/exit rules."""
        self._spec['entry_rules'] = entry_rules
        self._spec['exit_rules'] = exit_rules
        return self
    
    def with_holding_period(self, holding_period: int) -> "StrategyBuilder":
        """Set holding period."""
        self._spec['holding_period'] = holding_period
        return self
    
    def with_position_sizing(self, position_sizing: PositionSizing) -> "StrategyBuilder":
        """Set position sizing."""
        self._spec['position_sizing'] = position_sizing
        return self
    
    def with_cv_config(self, cv_config: CrossValidationConfig) -> "StrategyBuilder":
        """Set cross-validation configuration."""
        self._spec['cv_config'] = cv_config
        return self
    
    def build(self) -> StrategySpec:
        """Build and validate the StrategySpec."""
        return StrategySpec(**self._spec)
