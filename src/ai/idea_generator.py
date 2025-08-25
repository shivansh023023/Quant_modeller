"""
AI-powered idea generator for converting natural language to StrategySpec.

This module provides the main interface for generating structured trading
strategies from natural language descriptions using LLM integration.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

from ..strategies.schema import (
    StrategySpec, FeatureSpec, EntryExitRule, PositionSizing,
    CrossValidationConfig, FeatureType, TargetType, ValidationType
)
from ..strategies.validator import StrategyValidator
from .prompts import PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IdeaGenerator:
    """
    AI-powered generator for converting natural language to StrategySpec.
    
    This class provides methods to generate, validate, and improve
    trading strategies using LLM integration.
    """
    
    def __init__(self, llm_client=None, validation_enabled: bool = True):
        """
        Initialize the idea generator.
        
        Args:
            llm_client: LLM client for generating strategies (optional)
            validation_enabled: Whether to enable automatic validation
        """
        self.llm_client = llm_client
        self.validation_enabled = validation_enabled
        self.validator = StrategyValidator()
        self.prompts = PromptTemplates()
        
        # Default strategy parameters
        self.default_universe = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
        self.default_start_date = datetime(2020, 1, 1)
        self.default_end_date = datetime(2024, 1, 1)
    
    def generate_strategy(
        self, 
        natural_language_idea: str,
        universe: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        available_features: Optional[List[str]] = None
    ) -> StrategySpec:
        """
        Generate a StrategySpec from natural language description.
        
        Args:
            natural_language_idea: Natural language description of the strategy
            universe: Optional universe of tickers
            start_date: Optional start date
            end_date: Optional end date
            available_features: Optional list of available features
            
        Returns:
            Generated StrategySpec
        """
        logger.info(f"Generating strategy from: {natural_language_idea[:100]}...")
        
        # Use defaults if not provided
        universe = universe or self.default_universe
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        
        if self.llm_client:
            # Use LLM to generate strategy
            strategy_spec = self._generate_with_llm(
                natural_language_idea, universe, start_date, end_date, available_features
            )
        else:
            # Use template-based generation
            strategy_spec = self._generate_from_template(
                natural_language_idea, universe, start_date, end_date, available_features
            )
        
        # Validate the generated strategy
        if self.validation_enabled:
            is_valid, errors, warnings = self.validator.validate_strategy(strategy_spec)
            
            if not is_valid:
                logger.warning(f"Generated strategy has validation errors: {errors}")
                # Try to fix common issues
                strategy_spec = self._fix_validation_issues(strategy_spec, errors)
            
            if warnings:
                logger.info(f"Strategy validation warnings: {warnings}")
        
        logger.info(f"Generated strategy: {strategy_spec.name}")
        return strategy_spec
    
    def _generate_with_llm(
        self,
        natural_language_idea: str,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        available_features: Optional[List[str]]
    ) -> StrategySpec:
        """Generate strategy using LLM client."""
        try:
            # Generate prompt
            prompt = self.prompts.get_strategy_generation_prompt(
                natural_language_idea, available_features, universe
            )
            
            # Get LLM response
            response = self.llm_client.generate(prompt)
            
            # Parse JSON response
            strategy_dict = self._parse_llm_response(response)
            
            # Ensure required fields
            strategy_dict.update({
                'universe': universe,
                'start_date': start_date,
                'end_date': end_date
            })
            
            # Create StrategySpec
            return StrategySpec(**strategy_dict)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall back to template generation
            return self._generate_from_template(
                natural_language_idea, universe, start_date, end_date, available_features
            )
    
    def _generate_from_template(
        self,
        natural_language_idea: str,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        available_features: Optional[List[str]]
    ) -> StrategySpec:
        """Generate strategy using template-based approach."""
        # Extract key information from natural language
        strategy_info = self._extract_strategy_info(natural_language_idea)
        
        # Create default features if none specified
        if not strategy_info.get('features'):
            strategy_info['features'] = self._generate_default_features(
                strategy_info.get('strategy_type', 'mean_reversion')
            )
        
        # Create default rules if none specified
        if not strategy_info.get('entry_rules'):
            strategy_info['entry_rules'] = self._generate_default_entry_rules(
                strategy_info.get('strategy_type', 'mean_reversion')
            )
        
        if not strategy_info.get('exit_rules'):
            strategy_info['exit_rules'] = self._generate_default_exit_rules(
                strategy_info.get('strategy_type', 'mean_reversion')
            )
        
        # Build strategy spec
        strategy_dict = {
            'name': strategy_info.get('name', f"Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'description': natural_language_idea,
            'universe': universe,
            'start_date': start_date,
            'end_date': end_date,
            'features': strategy_info['features'],
            'target': strategy_info.get('target', TargetType.NEXT_DAY_RETURN),
            'target_lookback': strategy_info.get('target_lookback', 1),
            'entry_rules': strategy_info['entry_rules'],
            'exit_rules': strategy_info['exit_rules'],
            'holding_period': strategy_info.get('holding_period', 10),
            'position_sizing': strategy_info.get('position_sizing', self._default_position_sizing()),
            'max_positions': strategy_info.get('max_positions', 10),
            'cv_config': strategy_info.get('cv_config', self._default_cv_config())
        }
        
        return StrategySpec(**strategy_dict)
    
    def _extract_strategy_info(self, natural_language_idea: str) -> Dict[str, Any]:
        """Extract strategy information from natural language."""
        idea_lower = natural_language_idea.lower()
        
        # Determine strategy type
        strategy_type = 'mean_reversion'
        if any(word in idea_lower for word in ['momentum', 'trend', 'breakout']):
            strategy_type = 'momentum'
        elif any(word in idea_lower for word in ['arbitrage', 'pairs', 'correlation']):
            strategy_type = 'arbitrage'
        elif any(word in idea_lower for word in ['mean reversion', 'oversold', 'overbought']):
            strategy_type = 'mean_reversion'
        
        # Extract holding period
        holding_period = 10
        holding_match = re.search(r'hold.*?(\d+).*?day', idea_lower)
        if holding_match:
            holding_period = int(holding_match.group(1))
        
        # Extract lookback periods
        lookback_periods = [20, 50]
        lookback_matches = re.findall(r'(\d+).*?day', idea_lower)
        if lookback_matches:
            lookback_periods = [int(m) for m in lookback_matches[:2]]
        
        return {
            'strategy_type': strategy_type,
            'holding_period': holding_period,
            'lookback_periods': lookback_periods
        }
    
    def _generate_default_features(self, strategy_type: str) -> List[FeatureSpec]:
        """Generate default features based on strategy type."""
        if strategy_type == 'mean_reversion':
            features = [
                FeatureSpec(
                    name="rsi_14",
                    feature_type=FeatureType.TECHNICAL_INDICATOR,
                    lookback_period=14,
                    parameters={"period": 14}
                ),
                FeatureSpec(
                    name="bb_position_20",
                    feature_type=FeatureType.TECHNICAL_INDICATOR,
                    lookback_period=20,
                    parameters={"period": 20, "std_dev": 2}
                ),
                FeatureSpec(
                    name="volume_ratio_20",
                    feature_type=FeatureType.VOLUME_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                )
            ]
        elif strategy_type == 'momentum':
            features = [
                FeatureSpec(
                    name="momentum_20",
                    feature_type=FeatureType.PRICE_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                ),
                FeatureSpec(
                    name="volatility_20",
                    feature_type=FeatureType.PRICE_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                ),
                FeatureSpec(
                    name="volume_ratio_20",
                    feature_type=FeatureType.VOLUME_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                )
            ]
        else:
            # Generic features
            features = [
                FeatureSpec(
                    name="returns_20",
                    feature_type=FeatureType.PRICE_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                ),
                FeatureSpec(
                    name="volume_ratio_20",
                    feature_type=FeatureType.VOLUME_BASED,
                    lookback_period=20,
                    parameters={"period": 20}
                )
            ]
        
        return features
    
    def _generate_default_entry_rules(self, strategy_type: str) -> EntryExitRule:
        """Generate default entry rules based on strategy type."""
        if strategy_type == 'mean_reversion':
            condition = "rsi_14 < 30 and bb_position_20 < 0.1"
        elif strategy_type == 'momentum':
            condition = "momentum_20 > 0.05 and volume_ratio_20 > 1.2"
        else:
            condition = "returns_20 < -0.02 and volume_ratio_20 > 1.0"
        
        return EntryExitRule(
            condition=condition,
            lookback_period=20
        )
    
    def _generate_default_exit_rules(self, strategy_type: str) -> EntryExitRule:
        """Generate default exit rules based on strategy type."""
        if strategy_type == 'mean_reversion':
            condition = "rsi_14 > 70 or bb_position_20 > 0.9"
        elif strategy_type == 'momentum':
            condition = "momentum_20 < -0.02"
        else:
            condition = "returns_20 > 0.02"
        
        return EntryExitRule(
            condition=condition,
            lookback_period=20
        )
    
    def _default_position_sizing(self) -> PositionSizing:
        """Create default position sizing configuration."""
        return PositionSizing(
            method="volatility_target",
            volatility_target=0.15,
            max_position_size=0.2
        )
    
    def _default_cv_config(self) -> CrossValidationConfig:
        """Create default cross-validation configuration."""
        return CrossValidationConfig(
            validation_type=ValidationType.WALK_FORWARD,
            n_splits=5,
            train_size=0.7,
            purge_period=10,
            embargo_period=5
        )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract StrategySpec JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in LLM response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    def _fix_validation_issues(self, strategy_spec: StrategySpec, errors: List[str]) -> StrategySpec:
        """Attempt to fix common validation issues."""
        logger.info("Attempting to fix validation issues...")
        
        # Fix common issues
        for error in errors:
            if "lookback period" in error.lower():
                # Reduce lookback periods
                for feature in strategy_spec.features:
                    if feature.lookback_period > 100:
                        feature.lookback_period = 100
                
                if strategy_spec.entry_rules.lookback_period > 100:
                    strategy_spec.entry_rules.lookback_period = 100
                
                if strategy_spec.exit_rules.lookback_period > 100:
                    strategy_spec.exit_rules.lookback_period = 100
            
            elif "strategy period" in error.lower():
                # Extend strategy period
                strategy_spec.end_date = strategy_spec.start_date + timedelta(days=500)
        
        return strategy_spec
    
    def suggest_features(
        self, 
        strategy_description: str,
        existing_features: Optional[List[str]] = None
    ) -> List[FeatureSpec]:
        """
        Suggest additional features for a strategy.
        
        Args:
            strategy_description: Description of the strategy
            existing_features: List of already selected features
            
        Returns:
            List of suggested FeatureSpec objects
        """
        if self.llm_client:
            prompt = self.prompts.get_feature_suggestion_prompt(
                strategy_description, existing_features
            )
            
            try:
                response = self.llm_client.generate(prompt)
                # Parse response to extract features
                # This is a simplified implementation
                return self._parse_feature_suggestions(response)
            except Exception as e:
                logger.error(f"Feature suggestion failed: {e}")
        
        # Fall back to template-based suggestions
        return self._suggest_features_template(strategy_description, existing_features)
    
    def _suggest_features_template(
        self, 
        strategy_description: str,
        existing_features: Optional[List[str]]
    ) -> List[FeatureSpec]:
        """Generate feature suggestions using templates."""
        # Common technical indicators
        technical_features = [
            FeatureSpec(
                name="macd_12_26",
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                lookback_period=26,
                parameters={"fast": 12, "slow": 26, "signal": 9}
            ),
            FeatureSpec(
                name="stochastic_14",
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                lookback_period=14,
                parameters={"k_period": 14, "d_period": 3}
            ),
            FeatureSpec(
                name="atr_14",
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                lookback_period=14,
                parameters={"period": 14}
            )
        ]
        
        # Statistical features
        statistical_features = [
            FeatureSpec(
                name="returns_skew_20",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=20,
                parameters={"period": 20, "stat": "skew"}
            ),
            FeatureSpec(
                name="volatility_ratio_20_60",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=60,
                parameters={"short_period": 20, "long_period": 60}
            )
        ]
        
        # Volume features
        volume_features = [
            FeatureSpec(
                name="volume_ma_ratio_20",
                feature_type=FeatureType.VOLUME_BASED,
                lookback_period=20,
                parameters={"period": 20}
            ),
            FeatureSpec(
                name="volume_price_trend",
                feature_type=FeatureType.VOLUME_BASED,
                lookback_period=20,
                parameters={"period": 20}
            )
        ]
        
        # Combine features, avoiding duplicates
        all_features = technical_features + statistical_features + volume_features
        
        if existing_features:
            existing_names = [f.name for f in existing_features]
            all_features = [f for f in all_features if f.name not in existing_names]
        
        return all_features[:5]  # Return top 5 suggestions
    
    def _parse_feature_suggestions(self, response: str) -> List[FeatureSpec]:
        """Parse LLM response to extract feature suggestions."""
        # This is a simplified parser - in production you'd want more robust parsing
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                features_data = json.loads(json_match.group(0))
                features = []
                
                for feature_data in features_data:
                    try:
                        feature = FeatureSpec(**feature_data)
                        features.append(feature)
                    except Exception as e:
                        logger.warning(f"Failed to parse feature: {e}")
                        continue
                
                return features
        except Exception as e:
            logger.error(f"Failed to parse feature suggestions: {e}")
        
        # Return empty list if parsing fails
        return []
    
    def validate_strategy(self, strategy_spec: StrategySpec) -> Dict[str, Any]:
        """
        Validate a strategy specification.
        
        Args:
            strategy_spec: StrategySpec to validate
            
        Returns:
            Validation results
        """
        is_valid, errors, warnings = self.validator.validate_strategy(strategy_spec)
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "suggestions": self.validator.suggest_improvements(strategy_spec)
        }
    
    def improve_strategy(
        self, 
        strategy_spec: StrategySpec,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Get suggestions for improving a strategy.
        
        Args:
            strategy_spec: Current StrategySpec
            performance_metrics: Optional performance metrics
            
        Returns:
            List of improvement suggestions
        """
        if self.llm_client:
            prompt = self.prompts.get_strategy_improvement_prompt(
                strategy_spec.to_json(), performance_metrics
            )
            
            try:
                response = self.llm_client.generate(prompt)
                # Parse response to extract suggestions
                return self._parse_improvement_suggestions(response)
            except Exception as e:
                logger.error(f"Strategy improvement failed: {e}")
        
        # Fall back to template-based suggestions
        return self.validator.suggest_improvements(strategy_spec)
    
    def _parse_improvement_suggestions(self, response: str) -> List[str]:
        """Parse LLM response to extract improvement suggestions."""
        # Simple parsing - extract bullet points or numbered lists
        suggestions = []
        
        # Look for bullet points or numbered items
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                # Clean up the line
                suggestion = re.sub(r'^[•\-*\d\.\s]+', '', line).strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions if suggestions else ["Consider adding more features for better signal diversity"]
