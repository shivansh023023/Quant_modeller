"""
Model zoo for quantitative trading strategies.

This module provides a centralized collection of all available models
and utilities for model selection, comparison, and management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import logging
from pathlib import Path
import json

from .base_model import BaseModel
from .sklearn_models import (
    LinearRegressionModel, LogisticRegressionModel,
    RidgeRegressionModel, LassoRegressionModel,
    ElasticNetModel, SVRModel, SVCModel,
    KNNRegressorModel, KNNClassifierModel,
    MLPRegressorModel, MLPClassifierModel,
    SGDRegressorModel, SGDClassifierModel
)
from .tree_models import (
    RandomForestRegressorModel, RandomForestClassifierModel,
    GradientBoostingRegressorModel, GradientBoostingClassifierModel,
    ExtraTreesRegressorModel, ExtraTreesClassifierModel
)
from .ensemble_models import (
    VotingRegressorModel, VotingClassifierModel,
    BaggingRegressorModel, BaggingClassifierModel,
    StackingRegressorModel, StackingClassifierModel,
    CustomEnsembleModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelZoo:
    """
    Centralized collection of all available models.
    
    This class provides methods to create, manage, and compare
    different types of models for quantitative trading strategies.
    """
    
    def __init__(self):
        """Initialize the model zoo."""
        self.available_models = self._initialize_available_models()
        self.model_instances = {}
        self.model_metadata = {}
    
    def _initialize_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the dictionary of available models.
        
        Returns:
            Dictionary mapping model names to model information
        """
        models = {}
        
        # Linear models
        models['linear_regression'] = {
            'class': LinearRegressionModel,
            'type': 'regression',
            'category': 'linear',
            'description': 'Linear regression model',
            'default_params': {},
            'hyperparameter_ranges': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }
        
        models['logistic_regression'] = {
            'class': LogisticRegressionModel,
            'type': 'classification',
            'category': 'linear',
            'description': 'Logistic regression model',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
            }
        }
        
        models['ridge_regression'] = {
            'class': RidgeRegressionModel,
            'type': 'regression',
            'category': 'linear',
            'description': 'Ridge regression with L2 regularization',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }
        
        models['lasso_regression'] = {
            'class': LassoRegressionModel,
            'type': 'regression',
            'category': 'linear',
            'description': 'Lasso regression with L1 regularization',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }
        
        models['elastic_net'] = {
            'class': ElasticNetModel,
            'type': 'regression',
            'category': 'linear',
            'description': 'Elastic net regression with L1 and L2 regularization',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }
        
        # Support Vector models
        models['svr'] = {
            'class': SVRModel,
            'type': 'regression',
            'category': 'svm',
            'description': 'Support Vector Regression',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
            }
        }
        
        models['svc'] = {
            'class': SVCModel,
            'type': 'classification',
            'category': 'svm',
            'description': 'Support Vector Classification',
            'default_params': {'random_state': 42, 'probability': True},
            'hyperparameter_ranges': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
            }
        }
        
        # K-Nearest Neighbors
        models['knn_regressor'] = {
            'class': KNNRegressorModel,
            'type': 'regression',
            'category': 'neighbors',
            'description': 'K-Nearest Neighbors regression',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        models['knn_classifier'] = {
            'class': KNNClassifierModel,
            'type': 'classification',
            'category': 'neighbors',
            'description': 'K-Nearest Neighbors classification',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        # Neural Networks
        models['mlp_regressor'] = {
            'class': MLPRegressorModel,
            'type': 'regression',
            'category': 'neural_network',
            'description': 'Multi-layer Perceptron regression',
            'default_params': {'random_state': 42, 'max_iter': 1000},
            'hyperparameter_ranges': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive', 'invscaling']
            }
        }
        
        models['mlp_classifier'] = {
            'class': MLPClassifierModel,
            'type': 'classification',
            'category': 'neural_network',
            'description': 'Multi-layer Perceptron classification',
            'default_params': {'random_state': 42, 'max_iter': 1000},
            'hyperparameter_ranges': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive', 'invscaling']
            }
        }
        
        # Tree-based models
        models['random_forest_regressor'] = {
            'class': RandomForestRegressorModel,
            'type': 'regression',
            'category': 'tree',
            'description': 'Random Forest regression',
            'default_params': {'random_state': 42, 'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        models['random_forest_classifier'] = {
            'class': RandomForestClassifierModel,
            'type': 'classification',
            'category': 'tree',
            'description': 'Random Forest classification',
            'default_params': {'random_state': 42, 'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        models['gradient_boosting_regressor'] = {
            'class': GradientBoostingRegressorModel,
            'type': 'regression',
            'category': 'tree',
            'description': 'Gradient Boosting regression',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        models['gradient_boosting_classifier'] = {
            'class': GradientBoostingClassifierModel,
            'type': 'classification',
            'category': 'tree',
            'description': 'Gradient Boosting classification',
            'default_params': {'random_state': 42},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        models['extra_trees_regressor'] = {
            'class': ExtraTreesRegressorModel,
            'type': 'regression',
            'category': 'tree',
            'description': 'Extra Trees regression',
            'default_params': {'random_state': 42, 'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        models['extra_trees_classifier'] = {
            'class': ExtraTreesClassifierModel,
            'type': 'classification',
            'category': 'tree',
            'description': 'Extra Trees classification',
            'default_params': {'random_state': 42, 'n_jobs': -1},
            'hyperparameter_ranges': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        }
        
        # Ensemble models
        models['voting_regressor'] = {
            'class': VotingRegressorModel,
            'type': 'regression',
            'category': 'ensemble',
            'description': 'Voting regressor ensemble',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'weights': [None, 'uniform', 'custom']
            }
        }
        
        models['voting_classifier'] = {
            'class': VotingClassifierModel,
            'type': 'classification',
            'category': 'ensemble',
            'description': 'Voting classifier ensemble',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'voting': ['hard', 'soft'],
                'weights': [None, 'uniform', 'custom']
            }
        }
        
        models['bagging_regressor'] = {
            'class': BaggingRegressorModel,
            'type': 'regression',
            'category': 'ensemble',
            'description': 'Bagging regressor ensemble',
            'default_params': {'n_jobs': -1, 'random_state': 42},
            'hyperparameter_ranges': {
                'n_estimators': [10, 20, 50, 100],
                'max_samples': [0.5, 0.7, 0.9, 1.0],
                'max_features': [0.5, 0.7, 0.9, 1.0]
            }
        }
        
        models['bagging_classifier'] = {
            'class': BaggingClassifierModel,
            'type': 'classification',
            'category': 'ensemble',
            'description': 'Bagging classifier ensemble',
            'default_params': {'n_jobs': -1, 'random_state': 42},
            'hyperparameter_ranges': {
                'n_estimators': [10, 20, 50, 100],
                'max_samples': [0.5, 0.7, 0.9, 1.0],
                'max_features': [0.5, 0.7, 0.9, 1.0]
            }
        }
        
        models['stacking_regressor'] = {
            'class': StackingRegressorModel,
            'type': 'regression',
            'category': 'ensemble',
            'description': 'Stacking regressor ensemble',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'cv': [3, 5, 10],
                'stack_method': ['predict', 'predict_proba'],
                'passthrough': [True, False]
            }
        }
        
        models['stacking_classifier'] = {
            'class': StackingClassifierModel,
            'type': 'classification',
            'category': 'ensemble',
            'description': 'Stacking classifier ensemble',
            'default_params': {'n_jobs': -1},
            'hyperparameter_ranges': {
                'cv': [3, 5, 10],
                'stack_method': ['predict', 'predict_proba'],
                'passthrough': [True, False]
            }
        }
        
        return models
    
    def list_available_models(self, model_type: Optional[str] = None, category: Optional[str] = None) -> List[str]:
        """
        List available models, optionally filtered by type or category.
        
        Args:
            model_type: Filter by model type ('regression' or 'classification')
            category: Filter by model category
            
        Returns:
            List of available model names
        """
        available = []
        
        for name, info in self.available_models.items():
            if model_type and info['type'] != model_type:
                continue
            if category and info['category'] != category:
                continue
            available.append(name)
        
        return available
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found in zoo")
        
        return self.available_models[model_name].copy()
    
    def get_model(self, model_name: str, **kwargs) -> BaseModel:
        """
        Get (create) a model instance by name.
        
        Args:
            model_name: Name of the model
            **kwargs: Model parameters
            
        Returns:
            New model instance
        """
        return self.create_model(model_name, **kwargs)
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type of model ('regression' or 'classification')
            
        Returns:
            List of model names of the specified type
        """
        return [name for name, info in self.available_models.items() 
                if info['type'] == model_type]
    
    def get_models_by_category(self, category: str) -> List[str]:
        """
        Get all models of a specific category.
        
        Args:
            category: Category of model (e.g., 'linear', 'tree', 'ensemble')
            
        Returns:
            List of model names of the specified category
        """
        return [name for name, info in self.available_models.items() 
                if info['category'] == category]
    
    def create_model(self, model_name: str, **kwargs) -> BaseModel:
        """
        Create a new instance of a model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model parameters
            
        Returns:
            New model instance
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found in zoo")
        
        model_info = self.available_models[model_name]
        model_class = model_info['class']
        
        # Merge default parameters with provided parameters
        params = model_info['default_params'].copy()
        params.update(kwargs)
        
        try:
            model = model_class(**params)
            logger.info(f"Created model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise
    
    def create_ensemble(
        self, 
        base_models: List[Tuple[str, Any]], 
        ensemble_type: str = 'voting',
        **kwargs
    ) -> BaseModel:
        """
        Create an ensemble model.
        
        Args:
            base_models: List of (name, model) tuples
            ensemble_type: Type of ensemble ('voting', 'bagging', 'stacking')
            **kwargs: Additional ensemble parameters
            
        Returns:
            New ensemble model instance
        """
        if ensemble_type == 'voting':
            # Determine if regression or classification
            first_model = base_models[0][1]
            if hasattr(first_model, 'model_type'):
                model_type = first_model.model_type
            else:
                # Try to infer from model attributes
                if hasattr(first_model, 'predict_proba'):
                    model_type = 'classification'
                else:
                    model_type = 'regression'
            
            if model_type == 'regression':
                return VotingRegressorModel(base_models, **kwargs)
            else:
                return VotingClassifierModel(base_models, **kwargs)
        
        elif ensemble_type == 'bagging':
            base_estimator = base_models[0][1]
            if hasattr(base_estimator, 'model_type'):
                model_type = base_estimator.model_type
            else:
                model_type = 'regression'  # Default
            
            if model_type == 'regression':
                return BaggingRegressorModel(base_estimator, **kwargs)
            else:
                return BaggingClassifierModel(base_estimator, **kwargs)
        
        elif ensemble_type == 'stacking':
            # Determine if regression or classification
            first_model = base_models[0][1]
            if hasattr(first_model, 'model_type'):
                model_type = first_model.model_type
            else:
                model_type = 'regression'  # Default
            
            final_estimator = kwargs.pop('final_estimator', None)
            
            if model_type == 'regression':
                return StackingRegressorModel(base_models, final_estimator, **kwargs)
            else:
                return StackingClassifierModel(base_models, final_estimator, **kwargs)
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    def get_model_categories(self) -> Dict[str, List[str]]:
        """
        Get all model categories and their models.
        
        Returns:
            Dictionary mapping categories to lists of model names
        """
        categories = {}
        
        for name, info in self.available_models.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        return categories
    
    def get_model_types(self) -> Dict[str, List[str]]:
        """
        Get all model types and their models.
        
        Returns:
            Dictionary mapping types to lists of model names
        """
        types = {}
        
        for name, info in self.available_models.items():
            model_type = info['type']
            if model_type not in types:
                types[model_type] = []
            types[model_type].append(name)
        
        return types
    
    def get_hyperparameter_ranges(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Get hyperparameter ranges for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found in zoo")
        
        return self.available_models[model_name]['hyperparameter_ranges'].copy()
    
    def suggest_model_for_task(
        self, 
        task_type: str, 
        data_size: int, 
        feature_count: int,
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest models for a specific task.
        
        Args:
            task_type: Type of task ('regression' or 'classification')
            data_size: Number of training samples
            feature_count: Number of features
            requirements: Additional requirements (e.g., 'interpretable', 'fast')
            
        Returns:
            List of suggested model names
        """
        suggestions = []
        
        # Filter by task type
        available = self.list_available_models(model_type=task_type)
        
        # Score models based on requirements
        model_scores = []
        
        for model_name in available:
            score = 0
            info = self.available_models[model_name]
            
            # Data size considerations
            if data_size < 1000:
                if info['category'] in ['linear', 'svm']:
                    score += 2
                elif info['category'] == 'tree':
                    score += 1
            elif data_size < 10000:
                if info['category'] in ['tree', 'ensemble']:
                    score += 2
                elif info['category'] == 'linear':
                    score += 1
            else:
                if info['category'] in ['tree', 'ensemble']:
                    score += 2
                elif info['category'] == 'neural_network':
                    score += 1
            
            # Feature count considerations
            if feature_count < 10:
                if info['category'] == 'linear':
                    score += 2
                elif info['category'] == 'svm':
                    score += 1
            elif feature_count < 100:
                if info['category'] in ['tree', 'ensemble']:
                    score += 2
                elif info['category'] == 'linear':
                    score += 1
            else:
                if info['category'] in ['tree', 'ensemble', 'neural_network']:
                    score += 2
            
            # Requirements considerations
            if requirements:
                if requirements.get('interpretable', False):
                    if info['category'] == 'linear':
                        score += 3
                    elif info['category'] == 'tree':
                        score += 2
                
                if requirements.get('fast', False):
                    if info['category'] in ['linear', 'tree']:
                        score += 2
                    elif info['category'] == 'svm':
                        score += 1
                
                if requirements.get('robust', False):
                    if info['category'] in ['tree', 'ensemble']:
                        score += 2
                    elif info['category'] == 'linear':
                        score += 1
            
            model_scores.append((model_name, score))
        
        # Sort by score and return top suggestions
        model_scores.sort(key=lambda x: x[1], reverse=True)
        suggestions = [name for name, score in model_scores[:5]]
        
        return suggestions
    
    def save_model_zoo_info(self, filepath: str) -> bool:
        """
        Save model zoo information to a file.
        
        Args:
            filepath: Path to save the information
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Prepare data for saving
            zoo_info = {
                'available_models': self.available_models,
                'model_categories': self.get_model_categories(),
                'model_types': self.get_model_types(),
                'exported_at': pd.Timestamp.now().isoformat()
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(zoo_info, f, indent=2, default=str)
            
            logger.info(f"Model zoo info saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model zoo info: {e}")
            return False
    
    def load_model_zoo_info(self, filepath: str) -> bool:
        """
        Load model zoo information from a file.
        
        Args:
            filepath: Path to load the information from
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                zoo_info = json.load(f)
            
            # Update available models
            if 'available_models' in zoo_info:
                self.available_models.update(zoo_info['available_models'])
            
            logger.info(f"Model zoo info loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model zoo info: {e}")
            return False
    
    def get_zoo_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model zoo.
        
        Returns:
            Dictionary containing zoo summary information
        """
        categories = self.get_model_categories()
        types = self.get_model_types()
        
        summary = {
            'total_models': len(self.available_models),
            'categories': {cat: len(models) for cat, models in categories.items()},
            'types': {t: len(models) for t, models in types.items()},
            'model_names': list(self.available_models.keys())
        }
        
        return summary
