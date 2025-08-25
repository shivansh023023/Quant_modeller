"""
Tree-based models for quantitative trading strategies.

This module provides implementations of various tree-based models
including Random Forest, Gradient Boosting, and Extra Trees.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeModel(BaseModel):
    """
    Base class for tree-based models.
    
    This class provides common functionality for tree-based models
    while maintaining the BaseModel contract.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the tree model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('regression' or 'classification')
            **kwargs: Model parameters
        """
        super().__init__(model_name, model_type, **kwargs)
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> Any:
        """
        Create the underlying tree model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Tree model instance
        """
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'TreeModel':
        """
        Fit the tree model to the training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional validation data tuple (X_val, y_val)
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if not self.validate_input_data(X, y):
            raise ValueError("Invalid input data")
        
        # Set feature names
        self.set_feature_names(list(X.columns))
        self.set_target_name(y.name if y.name else 'target')
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Fit the model
        try:
            self.model.fit(X_processed, y, **kwargs)
            self.is_fitted = True
            
            # Calculate training metrics
            y_pred = self.predict(X_processed)
            self._calculate_training_metrics(y, y_pred)
            
            # Calculate validation metrics if provided
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                self._calculate_validation_metrics(y_val, y_val_pred)
            
            logger.info(f"Model {self.model_name} fitted successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            raise
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X_processed = self.preprocess_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support predict_proba")
        
        X_processed = self.preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _calculate_training_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Calculate and store training metrics."""
        if self.model_type == 'regression':
            self.training_history['mse'] = mean_squared_error(y_true, y_pred)
            self.training_history['rmse'] = np.sqrt(self.training_history['mse'])
            self.training_history['mae'] = mean_absolute_error(y_true, y_pred)
            self.training_history['r2'] = r2_score(y_true, y_pred)
        else:
            self.training_history['accuracy'] = accuracy_score(y_true, y_pred)
            self.training_history['precision'] = precision_score(y_true, y_pred, average='weighted')
            self.training_history['recall'] = recall_score(y_true, y_pred, average='weighted')
            self.training_history['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    def _calculate_validation_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Calculate and store validation metrics."""
        if self.model_type == 'regression':
            self.validation_metrics['val_mse'] = mean_squared_error(y_true, y_pred)
            self.validation_metrics['val_rmse'] = np.sqrt(self.validation_metrics['val_mse'])
            self.validation_metrics['val_mae'] = mean_absolute_error(y_true, y_pred)
            self.validation_metrics['val_r2'] = r2_score(y_true, y_pred)
        else:
            self.validation_metrics['val_accuracy'] = accuracy_score(y_true, y_pred)
            self.validation_metrics['val_precision'] = precision_score(y_true, y_pred, average='weighted')
            self.validation_metrics['val_recall'] = recall_score(y_true, y_pred, average='weighted')
            self.validation_metrics['val_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            logger.warning("Model does not support feature importance")
            return {}
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the trees in the model.
        
        Returns:
            Dictionary containing tree information
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        try:
            tree_info = {
                'n_estimators': len(self.model.estimators_),
                'n_features': self.model.n_features_in_,
                'feature_names': self.feature_names
            }
            
            # Get tree depths
            if hasattr(self.model, 'estimators_'):
                depths = [tree.tree_.max_depth for tree in self.model.estimators_]
                tree_info['tree_depths'] = {
                    'min': min(depths),
                    'max': max(depths),
                    'mean': np.mean(depths),
                    'std': np.std(depths)
                }
            
            return tree_info
            
        except Exception as e:
            logger.warning(f"Could not get tree info: {e}")
            return {'error': str(e)}
    
    def get_leaf_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the leaf indices for each sample.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of leaf indices
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting leaf samples")
        
        if not hasattr(self.model, 'apply'):
            raise ValueError("Model does not support apply method")
        
        X_processed = self.preprocess_features(X)
        return self.model.apply(X_processed)
    
    def get_decision_path(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the decision path for each sample.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of decision paths
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting decision paths")
        
        if not hasattr(self.model, 'decision_path'):
            raise ValueError("Model does not support decision_path method")
        
        X_processed = self.preprocess_features(X)
        return self.model.decision_path(X_processed).toarray()


class RandomForestRegressorModel(TreeModel):
    """Random Forest regression model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("random_forest_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest regressor."""
        return RandomForestRegressor(**kwargs)


class RandomForestClassifierModel(TreeModel):
    """Random Forest classification model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("random_forest_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(**kwargs)


class GradientBoostingRegressorModel(TreeModel):
    """Gradient Boosting regression model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("gradient_boosting_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> GradientBoostingRegressor:
        """Create Gradient Boosting regressor."""
        return GradientBoostingRegressor(**kwargs)
    
    def get_staged_predictions(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Get predictions for each stage of boosting.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of predictions for each stage
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting staged predictions")
        
        if not hasattr(self.model, 'staged_predict'):
            raise ValueError("Model does not support staged_predict method")
        
        X_processed = self.preprocess_features(X)
        return list(self.model.staged_predict(X_processed))


class GradientBoostingClassifierModel(TreeModel):
    """Gradient Boosting classification model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("gradient_boosting_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> GradientBoostingClassifier:
        """Create Gradient Boosting classifier."""
        return GradientBoostingClassifier(**kwargs)
    
    def get_staged_predictions(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Get predictions for each stage of boosting.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of predictions for each stage
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting staged predictions")
        
        if not hasattr(self.model, 'staged_predict'):
            raise ValueError("Model does not support staged_predict method")
        
        X_processed = self.preprocess_features(X)
        return list(self.model.staged_predict(X_processed))


class ExtraTreesRegressorModel(TreeModel):
    """Extra Trees regression model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("extra_trees_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> ExtraTreesRegressor:
        """Create Extra Trees regressor."""
        return ExtraTreesRegressor(**kwargs)


class ExtraTreesClassifierModel(TreeModel):
    """Extra Trees classification model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters if not provided
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("extra_trees_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> ExtraTreesClassifier:
        """Create Extra Trees classifier."""
        return ExtraTreesClassifier(**kwargs)
