"""
Tests for model zoo and AI functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.model_zoo import ModelZoo
from src.models.sklearn_models import SklearnModel
from src.ai.idea_generator import IdeaGenerator
from src.strategies.schema import StrategySpec, FeatureType, TargetType


class TestModelZoo:
    """Test model zoo functionality."""
    
    @pytest.fixture
    def model_zoo(self):
        """Create ModelZoo instance."""
        return ModelZoo()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some relationship to features
        y = pd.Series(
            X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1,
            name='target'
        )
        
        return X, y
    
    def test_model_zoo_initialization(self, model_zoo):
        """Test ModelZoo initialization."""
        assert model_zoo.available_models is not None
        assert len(model_zoo.available_models) > 0
        
        # Check for expected model categories
        model_names = list(model_zoo.available_models.keys())
        assert 'linear_regression' in model_names
        assert 'random_forest_regressor' in model_names or 'random_forest' in model_names
        assert 'logistic_regression' in model_names
    
    def test_get_model_by_name(self, model_zoo):
        """Test getting model by name."""
        model = model_zoo.get_model('linear_regression')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_get_models_by_type(self, model_zoo):
        """Test getting models by type."""
        regression_models = model_zoo.get_models_by_type('regression')
        classification_models = model_zoo.get_models_by_type('classification')
        
        assert len(regression_models) > 0
        assert len(classification_models) > 0
        
        # Check that all returned models have correct type
        for model_name in regression_models:
            model_info = model_zoo.available_models[model_name]
            assert model_info['type'] == 'regression'
    
    def test_get_models_by_category(self, model_zoo):
        """Test getting models by category."""
        linear_models = model_zoo.get_models_by_category('linear')
        
        assert len(linear_models) > 0
        
        # Check that all returned models have correct category
        for model_name in linear_models:
            model_info = model_zoo.available_models[model_name]
            assert model_info['category'] == 'linear'
    
    def test_model_training_and_prediction(self, model_zoo, sample_training_data):
        """Test model training and prediction."""
        X, y = sample_training_data
        
        # Get a simple model
        model = model_zoo.get_model('linear_regression')
        
        # Train the model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
        
        # Check that model learned something reasonable
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.3  # Should have some predictive power
    
    def test_model_validation_metrics(self, model_zoo, sample_training_data):
        """Test model validation metrics."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = model_zoo.get_model('linear_regression')
        model.fit(X_train, y_train, validation_data=(X_test, y_test))
        
        # Check that model has metrics
        assert hasattr(model, 'training_metrics')
        assert hasattr(model, 'validation_metrics')
        
        # Validate metric keys
        if hasattr(model, 'training_metrics') and model.training_metrics:
            assert 'mse' in model.training_metrics or 'r2' in model.training_metrics


class TestSklearnModel:
    """Test SklearnModel wrapper."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for model testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = pd.Series(
            X['feature_0'] + 0.5 * X['feature_1'] + np.random.randn(n_samples) * 0.1,
            name='target'
        )
        
        return X, y
    
    def test_sklearn_model_creation(self):
        """Test SklearnModel creation."""
        model = SklearnModel('linear_regression', 'regression')
        
        assert model.model_name == 'linear_regression'
        assert model.model_type == 'regression'
        assert model.model is not None
    
    def test_sklearn_model_fitting(self, sample_data):
        """Test SklearnModel fitting."""
        X, y = sample_data
        
        model = SklearnModel('linear_regression', 'regression')
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.feature_names == list(X.columns)
        assert model.target_name == 'target'
    
    def test_sklearn_model_prediction(self, sample_data):
        """Test SklearnModel prediction."""
        X, y = sample_data
        
        model = SklearnModel('linear_regression', 'regression')
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
    
    def test_sklearn_classification_model(self, sample_data):
        """Test SklearnModel with classification."""
        X, y = sample_data
        
        # Convert to classification problem
        y_binary = (y > y.median()).astype(int)
        
        model = SklearnModel('logistic_regression', 'classification')
        model.fit(X, y_binary)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(y_binary)
        assert probabilities.shape == (len(y_binary), 2)  # Binary classification
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_invalid_model_type(self):
        """Test invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            SklearnModel('test_model', 'invalid_type')


class TestIdeaGenerator:
    """Test AI idea generation functionality."""
    
    @pytest.fixture
    def idea_generator(self):
        """Create IdeaGenerator instance without LLM client."""
        return IdeaGenerator(llm_client=None, validation_enabled=True)
    
    def test_idea_generator_initialization(self, idea_generator):
        """Test IdeaGenerator initialization."""
        assert idea_generator.llm_client is None
        assert idea_generator.validation_enabled is True
        assert idea_generator.validator is not None
        assert idea_generator.prompts is not None
    
    def test_strategy_generation_from_template(self, idea_generator):
        """Test strategy generation using templates."""
        natural_language_idea = "Buy stocks when RSI is below 30 and hold for 5 days"
        
        strategy = idea_generator.generate_strategy(
            natural_language_idea,
            universe=["AAPL", "GOOGL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        assert isinstance(strategy, StrategySpec)
        assert strategy.description == natural_language_idea
        assert strategy.universe == ["AAPL", "GOOGL"]
        assert len(strategy.features) > 0
        assert strategy.entry_rules is not None
        assert strategy.exit_rules is not None
    
    def test_strategy_type_detection(self, idea_generator):
        """Test strategy type detection from natural language."""
        # Test mean reversion detection
        mean_reversion_idea = "Buy oversold stocks when they are mean reverting"
        strategy_info = idea_generator._extract_strategy_info(mean_reversion_idea)
        assert strategy_info['strategy_type'] == 'mean_reversion'
        
        # Test momentum detection
        momentum_idea = "Follow the trend and buy stocks with strong momentum"
        strategy_info = idea_generator._extract_strategy_info(momentum_idea)
        assert strategy_info['strategy_type'] == 'momentum'
        
        # Test arbitrage detection
        arbitrage_idea = "Exploit correlation between pairs of stocks"
        strategy_info = idea_generator._extract_strategy_info(arbitrage_idea)
        assert strategy_info['strategy_type'] == 'arbitrage'
    
    def test_holding_period_extraction(self, idea_generator):
        """Test holding period extraction from natural language."""
        # Test explicit holding period
        idea_with_period = "Buy stocks and hold for 15 days"
        strategy_info = idea_generator._extract_strategy_info(idea_with_period)
        assert strategy_info['holding_period'] == 15
        
        # Test default holding period
        idea_without_period = "Buy undervalued stocks"
        strategy_info = idea_generator._extract_strategy_info(idea_without_period)
        assert strategy_info['holding_period'] == 10  # Default
    
    def test_default_feature_generation(self, idea_generator):
        """Test default feature generation for different strategy types."""
        # Test mean reversion features
        mr_features = idea_generator._generate_default_features('mean_reversion')
        feature_names = [f.name for f in mr_features]
        assert 'rsi_14' in feature_names
        assert 'bb_position_20' in feature_names
        
        # Test momentum features
        momentum_features = idea_generator._generate_default_features('momentum')
        feature_names = [f.name for f in momentum_features]
        assert 'momentum_20' in feature_names
        assert 'volatility_20' in feature_names
    
    def test_default_rules_generation(self, idea_generator):
        """Test default rule generation for different strategy types."""
        # Test mean reversion rules
        mr_entry = idea_generator._generate_default_entry_rules('mean_reversion')
        assert 'rsi_14 < 30' in mr_entry.condition
        
        mr_exit = idea_generator._generate_default_exit_rules('mean_reversion')
        assert 'rsi_14 > 70' in mr_exit.condition
        
        # Test momentum rules
        momentum_entry = idea_generator._generate_default_entry_rules('momentum')
        assert 'momentum_20 > 0.05' in momentum_entry.condition
        
        momentum_exit = idea_generator._generate_default_exit_rules('momentum')
        assert 'momentum_20 < -0.02' in momentum_exit.condition
    
    def test_feature_suggestions(self, idea_generator):
        """Test feature suggestion functionality."""
        strategy_description = "A momentum strategy based on price trends"
        
        suggested_features = idea_generator.suggest_features(strategy_description)
        
        assert isinstance(suggested_features, list)
        assert len(suggested_features) > 0
        
        # Should return FeatureSpec objects
        for feature in suggested_features:
            assert hasattr(feature, 'name')
            assert hasattr(feature, 'feature_type')
            assert hasattr(feature, 'lookback_period')
    
    def test_strategy_validation(self, idea_generator):
        """Test strategy validation."""
        # Create a valid strategy
        valid_strategy = StrategySpec(
            name="Test Strategy",
            description="A test strategy",
            universe=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            features=[
                {
                    "name": "test_feature",
                    "feature_type": FeatureType.PRICE_BASED,
                    "lookback_period": 20,
                    "parameters": {}
                }
            ],
            target=TargetType.NEXT_DAY_RETURN,
            entry_rules={"condition": "test_feature > 0", "lookback_period": 10},
            exit_rules={"condition": "test_feature < 0", "lookback_period": 10},
            holding_period=5,
            position_sizing={"method": "fixed_size"},
            cv_config={"validation_type": "walk_forward"}
        )
        
        validation_result = idea_generator.validate_strategy(valid_strategy)
        
        assert isinstance(validation_result, dict)
        assert 'is_valid' in validation_result
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        assert 'suggestions' in validation_result
    
    def test_strategy_improvement_suggestions(self, idea_generator):
        """Test strategy improvement suggestions."""
        # Create a simple strategy
        simple_strategy = StrategySpec(
            name="Simple Strategy",
            description="A very simple strategy",
            universe=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            features=[
                {
                    "name": "simple_feature",
                    "feature_type": FeatureType.PRICE_BASED,
                    "lookback_period": 10,
                    "parameters": {}
                }
            ],
            target=TargetType.NEXT_DAY_RETURN,
            entry_rules={"condition": "simple_feature > 0", "lookback_period": 5},
            exit_rules={"condition": "simple_feature < 0", "lookback_period": 5},
            holding_period=5,
            position_sizing={"method": "fixed_size"},
            cv_config={"validation_type": "walk_forward"}
        )
        
        # Mock performance metrics
        performance_metrics = {
            "sharpe_ratio": 0.5,  # Low Sharpe ratio
            "max_drawdown": -0.3  # High drawdown
        }
        
        suggestions = idea_generator.improve_strategy(simple_strategy, performance_metrics)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should provide meaningful suggestions
        suggestions_text = " ".join(suggestions).lower()
        assert any(word in suggestions_text for word in ['feature', 'improve', 'add', 'risk'])


class TestModelIntegration:
    """Test model integration and workflow."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'momentum': np.random.randn(n_samples),
            'volatility': np.abs(np.random.randn(n_samples)),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples)
        })
        
        # Create target with realistic relationship
        y = pd.Series(
            0.3 * X['momentum'] - 0.2 * X['volatility'] + 0.1 * np.log(X['volume_ratio']) + 
            np.random.randn(n_samples) * 0.05,
            name='next_day_return'
        )
        
        return X, y
    
    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'rsi': np.random.uniform(0, 100, n_samples),
            'bb_position': np.random.uniform(0, 1, n_samples),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples)
        })
        
        # Create binary target (buy/sell signal)
        y = pd.Series(
            ((X['rsi'] < 30) & (X['bb_position'] < 0.2) & (X['volume_ratio'] > 1.2)).astype(int),
            name='buy_signal'
        )
        
        return X, y
    
    def test_regression_model_workflow(self, regression_data):
        """Test complete regression model workflow."""
        X, y = regression_data
        model_zoo = ModelZoo()
        
        # Test multiple regression models
        regression_models = ['linear_regression', 'ridge_regression']
        
        for model_name in regression_models:
            if model_name in model_zoo.available_models:
                model = model_zoo.get_model(model_name)
                
                # Train
                model.fit(X, y)
                
                # Predict
                predictions = model.predict(X)
                
                # Validate
                assert len(predictions) == len(y)
                assert not np.isnan(predictions).any()
                
                # Check reasonable performance
                mse = np.mean((y - predictions) ** 2)
                assert mse < 1.0  # Should be reasonably accurate
    
    def test_classification_model_workflow(self, classification_data):
        """Test complete classification model workflow."""
        X, y = classification_data
        model_zoo = ModelZoo()
        
        # Test classification models
        classification_models = ['logistic_regression']
        
        for model_name in classification_models:
            if model_name in model_zoo.available_models:
                model = model_zoo.get_model(model_name)
                
                # Train
                model.fit(X, y)
                
                # Predict
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                
                # Validate predictions
                assert len(predictions) == len(y)
                assert np.all(np.isin(predictions, [0, 1]))  # Binary predictions
                
                # Validate probabilities
                assert probabilities.shape == (len(y), 2)
                assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
                
                # Check reasonable performance
                accuracy = np.mean(predictions == y)
                assert accuracy > 0.4  # Should be better than random for this data
