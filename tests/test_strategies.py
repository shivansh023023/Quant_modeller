"""
Tests for strategy schema and validation.
"""

import pytest
from datetime import datetime, timedelta
from src.strategies.schema import (
    StrategySpec, FeatureSpec, EntryExitRule, PositionSizing, 
    CrossValidationConfig, FeatureType, TargetType, ValidationType
)


class TestFeatureSpec:
    """Test FeatureSpec validation."""
    
    def test_valid_feature_spec(self):
        """Test creating a valid feature spec."""
        feature = FeatureSpec(
            name="rsi_14",
            feature_type=FeatureType.TECHNICAL_INDICATOR,
            lookback_period=14,
            parameters={"period": 14}
        )
        
        assert feature.name == "rsi_14"
        assert feature.feature_type == FeatureType.TECHNICAL_INDICATOR
        assert feature.lookback_period == 14
        assert feature.parameters == {"period": 14}
    
    def test_invalid_lookback_period(self):
        """Test validation of invalid lookback period."""
        with pytest.raises(ValueError, match="Lookback period cannot exceed 252 days"):
            FeatureSpec(
                name="invalid_feature",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=300,  # Too long
                parameters={}
            )
    
    def test_zero_lookback_period(self):
        """Test validation of zero lookback period."""
        with pytest.raises(ValueError):
            FeatureSpec(
                name="invalid_feature",
                feature_type=FeatureType.PRICE_BASED,
                lookback_period=0,  # Should be > 0
                parameters={}
            )


class TestEntryExitRule:
    """Test EntryExitRule validation."""
    
    def test_valid_rule(self):
        """Test creating a valid rule."""
        rule = EntryExitRule(
            condition="rsi_14 < 30",
            lookback_period=20
        )
        
        assert rule.condition == "rsi_14 < 30"
        assert rule.lookback_period == 20
    
    def test_invalid_lookback_period(self):
        """Test validation of invalid lookback period."""
        with pytest.raises(ValueError, match="Rule lookback period cannot exceed 252 days"):
            EntryExitRule(
                condition="rsi_14 < 30",
                lookback_period=300  # Too long
            )


class TestPositionSizing:
    """Test PositionSizing validation."""
    
    def test_valid_position_sizing(self):
        """Test creating valid position sizing."""
        ps = PositionSizing(
            method="volatility_target",
            volatility_target=0.15,
            max_position_size=0.2
        )
        
        assert ps.method == "volatility_target"
        assert ps.volatility_target == 0.15
        assert ps.max_position_size == 0.2
    
    def test_invalid_method(self):
        """Test validation of invalid method."""
        with pytest.raises(ValueError, match="Position sizing method must be one of"):
            PositionSizing(method="invalid_method")
    
    def test_invalid_volatility_target(self):
        """Test validation of invalid volatility target."""
        with pytest.raises(ValueError):
            PositionSizing(
                method="volatility_target",
                volatility_target=0.6  # Too high (> 0.5)
            )


class TestStrategySpec:
    """Test complete StrategySpec validation."""
    
    @pytest.fixture
    def valid_strategy_spec(self):
        """Create a valid strategy spec for testing."""
        return StrategySpec(
            name="Test Strategy",
            description="A test strategy",
            universe=["AAPL", "GOOGL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            features=[
                FeatureSpec(
                    name="rsi_14",
                    feature_type=FeatureType.TECHNICAL_INDICATOR,
                    lookback_period=14,
                    parameters={"period": 14}
                )
            ],
            target=TargetType.NEXT_DAY_RETURN,
            entry_rules=EntryExitRule(
                condition="rsi_14 < 30",
                lookback_period=14
            ),
            exit_rules=EntryExitRule(
                condition="rsi_14 > 70",
                lookback_period=14
            ),
            holding_period=10,
            position_sizing=PositionSizing(
                method="fixed_size",
                max_position_size=0.2
            ),
            cv_config=CrossValidationConfig(
                validation_type=ValidationType.WALK_FORWARD,
                n_splits=5,
                train_size=0.7
            )
        )
    
    def test_valid_strategy_creation(self, valid_strategy_spec):
        """Test creating a valid strategy."""
        strategy = valid_strategy_spec
        
        assert strategy.name == "Test Strategy"
        assert len(strategy.universe) == 2
        assert len(strategy.features) == 1
        assert strategy.target == TargetType.NEXT_DAY_RETURN
    
    def test_date_validation(self):
        """Test date validation."""
        with pytest.raises(ValueError, match="Start date must be before end date"):
            StrategySpec(
                name="Invalid Strategy",
                description="Test",
                universe=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2022, 1, 1),  # End before start
                features=[
                    FeatureSpec(
                        name="test_feature",
                        feature_type=FeatureType.PRICE_BASED,
                        lookback_period=10,
                        parameters={}
                    )
                ],
                target=TargetType.NEXT_DAY_RETURN,
                entry_rules=EntryExitRule(condition="test_feature > 0", lookback_period=5),
                exit_rules=EntryExitRule(condition="test_feature < 0", lookback_period=5),
                holding_period=5,
                position_sizing=PositionSizing(method="fixed_size"),
                cv_config=CrossValidationConfig(validation_type=ValidationType.WALK_FORWARD)
            )
    
    def test_minimum_period_validation(self):
        """Test minimum strategy period validation."""
        with pytest.raises(ValueError, match="Strategy period must be at least 1 trading year"):
            StrategySpec(
                name="Short Strategy",
                description="Test",
                universe=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 1),  # Too short
                features=[
                    FeatureSpec(
                        name="test_feature",
                        feature_type=FeatureType.PRICE_BASED,
                        lookback_period=10,
                        parameters={}
                    )
                ],
                target=TargetType.NEXT_DAY_RETURN,
                entry_rules=EntryExitRule(condition="test_feature > 0", lookback_period=5),
                exit_rules=EntryExitRule(condition="test_feature < 0", lookback_period=5),
                holding_period=5,
                position_sizing=PositionSizing(method="fixed_size"),
                cv_config=CrossValidationConfig(validation_type=ValidationType.WALK_FORWARD)
            )
    
    def test_feature_lookback_validation(self):
        """Test feature lookback validation."""
        with pytest.raises(ValueError, match="Feature.*lookback period.*must be less than strategy period"):
            StrategySpec(
                name="Invalid Lookback Strategy",
                description="Test",
                universe=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 9, 10),  # Exactly 252 days
                features=[
                    FeatureSpec(
                        name="invalid_feature",
                        feature_type=FeatureType.PRICE_BASED,
                        lookback_period=252,  # Equal to strategy period (should trigger >=)
                        parameters={}
                    )
                ],
                target=TargetType.NEXT_DAY_RETURN,
                entry_rules=EntryExitRule(condition="invalid_feature > 0", lookback_period=5),
                exit_rules=EntryExitRule(condition="invalid_feature < 0", lookback_period=5),
                holding_period=5,
                position_sizing=PositionSizing(method="fixed_size"),
                cv_config=CrossValidationConfig(validation_type=ValidationType.WALK_FORWARD)
            )
    
    def test_max_lookback_calculation(self, valid_strategy_spec):
        """Test max lookback calculation."""
        max_lookback = valid_strategy_spec.get_max_lookback()
        assert max_lookback == 14  # Max of feature (14) and rules (14)
    
    def test_serialization(self, valid_strategy_spec):
        """Test JSON serialization/deserialization."""
        json_str = valid_strategy_spec.to_json()
        reconstructed = StrategySpec.from_json(json_str)
        
        assert reconstructed.name == valid_strategy_spec.name
        assert reconstructed.universe == valid_strategy_spec.universe
        assert len(reconstructed.features) == len(valid_strategy_spec.features)
