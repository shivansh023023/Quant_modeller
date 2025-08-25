"""
Strategy validation utilities.

This module provides validation functions to ensure strategies
comply with best practices and prevent data leakage.
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .schema import StrategySpec, FeatureSpec, EntryExitRule


class StrategyValidator:
    """Validator for StrategySpec instances."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_strategy(self, strategy: StrategySpec) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete strategy specification.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_basic_fields(strategy)
        self._validate_data_leakage(strategy)
        self._validate_feature_engineering(strategy)
        self._validate_risk_management(strategy)
        self._validate_validation_setup(strategy)
        self._validate_trading_rules(strategy)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_basic_fields(self, strategy: StrategySpec):
        """Validate basic strategy fields."""
        # Check universe size
        if len(strategy.universe) > 100:
            self.warnings.append(
                f"Large universe ({len(strategy.universe)} symbols) may impact performance"
            )
        
        # Check date range
        date_range = (strategy.end_date - strategy.start_date).days
        if date_range < 252:
            self.errors.append(
                f"Strategy period ({date_range} days) is too short. "
                "Minimum 252 days (1 trading year) required."
            )
        
        # Check feature count
        if len(strategy.features) > 50:
            self.warnings.append(
                f"Many features ({len(strategy.features)}) may lead to overfitting"
            )
    
    def _validate_data_leakage(self, strategy: StrategySpec):
        """Validate that no data leakage exists."""
        max_lookback = strategy.get_max_lookback()
        
        # Check that features don't use future information
        for feature in strategy.features:
            if feature.lookback_period > max_lookback:
                self.errors.append(
                    f"Feature {feature.name} lookback ({feature.lookback_period}) "
                    f"exceeds maximum allowed ({max_lookback})"
                )
        
        # Check entry/exit rules
        if strategy.entry_rules.lookback_period > max_lookback:
            self.errors.append(
                f"Entry rule lookback ({strategy.entry_rules.lookback_period}) "
                f"exceeds maximum allowed ({max_lookback})"
            )
        
        if strategy.exit_rules.lookback_period > max_lookback:
            self.errors.append(
                f"Exit rule lookback ({strategy.exit_rules.lookback_period}) "
                f"exceeds maximum allowed ({max_lookback})"
            )
        
        # Check target lookback
        if strategy.target_lookback > 5:
            self.warnings.append(
                f"Target lookback ({strategy.target_lookback}) is large. "
                "Consider if this makes sense for your strategy."
            )
    
    def _validate_feature_engineering(self, strategy: StrategySpec):
        """Validate feature engineering choices."""
        feature_types = [f.feature_type for f in strategy.features]
        
        # Check for feature diversity
        if len(set(feature_types)) < 2:
            self.warnings.append(
                "Low feature diversity. Consider mixing different feature types."
            )
        
        # Check for potential multicollinearity
        price_based_count = sum(1 for ft in feature_types if ft.value == "price_based")
        if price_based_count > len(strategy.features) * 0.7:
            self.warnings.append(
                "High proportion of price-based features may lead to multicollinearity"
            )
        
        # Check lookback period distribution
        lookbacks = [f.lookback_period for f in strategy.features]
        if max(lookbacks) - min(lookbacks) < 10:
            self.warnings.append(
                "Similar lookback periods across features may reduce signal diversity"
            )
    
    def _validate_risk_management(self, strategy: StrategySpec):
        """Validate risk management parameters."""
        # Check position sizing
        if strategy.position_sizing.method == "volatility_target":
            if not strategy.position_sizing.volatility_target:
                self.errors.append(
                    "Volatility target method requires volatility_target parameter"
                )
        
        if strategy.position_sizing.method == "risk_per_trade":
            if not strategy.position_sizing.risk_per_trade:
                self.errors.append(
                    "Risk per trade method requires risk_per_trade parameter"
                )
        
        # Check stop loss and take profit
        if strategy.stop_loss and strategy.take_profit:
            if strategy.stop_loss >= strategy.take_profit:
                self.errors.append(
                    "Stop loss must be less than take profit"
                )
        
        # Check max positions
        if strategy.max_positions > len(strategy.universe):
            self.warnings.append(
                f"Max positions ({strategy.max_positions}) exceeds universe size "
                f"({len(strategy.universe)})"
            )
    
    def _validate_validation_setup(self, strategy: StrategySpec):
        """Validate cross-validation configuration."""
        cv_config = strategy.cv_config
        
        if cv_config.validation_type.value == "walk_forward":
            if cv_config.n_splits < 3:
                self.warnings.append(
                    "Walk-forward validation with < 3 splits may not be robust"
                )
        
        if cv_config.validation_type.value == "purged_kfold":
            if cv_config.purge_period < 5:
                self.warnings.append(
                    "Purge period < 5 days may not fully prevent data leakage"
                )
        
        # Check train size
        if cv_config.train_size < 0.6:
            self.warnings.append(
                f"Small training set ({cv_config.train_size:.1%}) may lead to overfitting"
            )
    
    def _validate_trading_rules(self, strategy: StrategySpec):
        """Validate trading rule logic."""
        # Check holding period vs lookback periods
        if strategy.holding_period < max(
            strategy.entry_rules.lookback_period,
            strategy.exit_rules.lookback_period
        ):
            self.warnings.append(
                "Holding period may be too short relative to rule lookbacks"
            )
        
        # Check for potential rapid trading
        if strategy.holding_period < 5:
            self.warnings.append(
                "Very short holding period may lead to excessive trading costs"
            )
    
    def validate_feature_expression(self, expression: str, available_features: List[str]) -> bool:
        """
        Validate that a feature expression only uses available features.
        
        Args:
            expression: String expression to validate
            available_features: List of available feature names
            
        Returns:
            True if expression is valid, False otherwise
        """
        # Simple validation - check if all referenced features exist
        # This is a basic check; in production you might want more sophisticated parsing
        
        for feature in available_features:
            if feature in expression:
                expression = expression.replace(feature, "")
        
        # Remove common operators and functions
        import re
        expression = re.sub(r'[\+\-\*\/\(\)\<\>\=\!\&\|]', '', expression)
        expression = re.sub(r'\b(and|or|not)\b', '', expression)
        expression = re.sub(r'\b(if|else|elif)\b', '', expression)
        
        # Check if any non-whitespace characters remain
        if expression.strip():
            return False
        
        return True
    
    def suggest_improvements(self, strategy: StrategySpec) -> List[str]:
        """Suggest improvements for the strategy."""
        suggestions = []
        
        # Feature engineering suggestions
        if len(strategy.features) < 5:
            suggestions.append(
                "Consider adding more features for better signal diversity"
            )
        
        # Risk management suggestions
        if not strategy.stop_loss:
            suggestions.append(
                "Consider adding stop-loss rules for risk management"
            )
        
        # Validation suggestions
        if strategy.cv_config.validation_type.value == "walk_forward":
            if strategy.cv_config.n_splits < 5:
                suggestions.append(
                    "Increase walk-forward splits for more robust validation"
                )
        
        return suggestions


def quick_validate(strategy: StrategySpec) -> bool:
    """
    Quick validation check for a strategy.
    
    Args:
        strategy: StrategySpec to validate
        
    Returns:
        True if strategy is valid, False otherwise
    """
    validator = StrategyValidator()
    is_valid, _, _ = validator.validate_strategy(strategy)
    return is_valid


def get_validation_report(strategy: StrategySpec) -> Dict[str, Any]:
    """
    Get a comprehensive validation report for a strategy.
    
    Args:
        strategy: StrategySpec to validate
        
    Returns:
        Dictionary containing validation results and suggestions
    """
    validator = StrategyValidator()
    is_valid, errors, warnings = validator.validate_strategy(strategy)
    suggestions = validator.suggest_improvements(strategy)
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "max_lookback": strategy.get_max_lookback(),
        "feature_count": len(strategy.features),
        "universe_size": len(strategy.universe),
        "period_days": (strategy.end_date - strategy.start_date).days
    }
