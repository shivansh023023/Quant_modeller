"""
Ensemble models for quantitative trading strategies.

This module provides implementations of various ensemble models
including voting, stacking, and bagging approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.ensemble import (
    VotingRegressor, VotingClassifier,
    BaggingRegressor, BaggingClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Base class for ensemble models.
    
    This class provides common functionality for ensemble models
    while maintaining the BaseModel contract.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the ensemble model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('regression' or 'classification')
            **kwargs: Model parameters
        """
        super().__init__(model_name, model_type, **kwargs)
        self.base_models = kwargs.get('base_models', [])
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> Any:
        """
        Create the underlying ensemble model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Ensemble model instance
        """
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'EnsembleModel':
        """
        Fit the ensemble model to the training data.
        
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
    
    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting base predictions")
        
        X_processed = self.preprocess_features(X)
        base_predictions = {}
        
        try:
            if hasattr(self.model, 'estimators_'):
                for i, estimator in enumerate(self.model.estimators_):
                    if hasattr(estimator, 'predict'):
                        pred = estimator.predict(X_processed)
                        base_predictions[f'base_model_{i}'] = pred
        except Exception as e:
            logger.warning(f"Could not get base model predictions: {e}")
        
        return base_predictions
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each base model.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_fitted:
            return {}
        
        try:
            if hasattr(self.model, 'weights_'):
                return dict(zip(range(len(self.model.weights_)), self.model.weights_))
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not get ensemble weights: {e}")
            return {}


class VotingRegressorModel(EnsembleModel):
    """Voting regressor ensemble model wrapper."""
    
    def __init__(self, base_models: List[Tuple[str, Any]], **kwargs):
        """
        Initialize voting regressor.
        
        Args:
            base_models: List of (name, model) tuples
            **kwargs: Additional parameters
        """
        self.base_models = base_models
        default_params = {
            'estimators': base_models,
            'weights': None,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        super().__init__("voting_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> VotingRegressor:
        """Create Voting Regressor."""
        return VotingRegressor(**kwargs)


class VotingClassifierModel(EnsembleModel):
    """Voting classifier ensemble model wrapper."""
    
    def __init__(self, base_models: List[Tuple[str, Any]], **kwargs):
        """
        Initialize voting classifier.
        
        Args:
            base_models: List of (name, model) tuples
            **kwargs: Additional parameters
        """
        self.base_models = base_models
        default_params = {
            'estimators': base_models,
            'voting': 'soft',
            'weights': None,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        super().__init__("voting_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> VotingClassifier:
        """Create Voting Classifier."""
        return VotingClassifier(**kwargs)


class BaggingRegressorModel(EnsembleModel):
    """Bagging regressor ensemble model wrapper."""
    
    def __init__(self, base_estimator: Any, **kwargs):
        """
        Initialize bagging regressor.
        
        Args:
            base_estimator: Base estimator to use
            **kwargs: Additional parameters
        """
        self.base_estimator = base_estimator
        default_params = {
            'base_estimator': base_estimator,
            'n_estimators': 10,
            'max_samples': 1.0,
            'max_features': 1.0,
            'bootstrap': True,
            'bootstrap_features': False,
            'n_jobs': -1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("bagging_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> BaggingRegressor:
        """Create Bagging Regressor."""
        return BaggingRegressor(**kwargs)


class BaggingClassifierModel(EnsembleModel):
    """Bagging classifier ensemble model wrapper."""
    
    def __init__(self, base_estimator: Any, **kwargs):
        """
        Initialize bagging classifier.
        
        Args:
            base_estimator: Base estimator to use
            **kwargs: Additional parameters
        """
        self.base_estimator = base_estimator
        default_params = {
            'base_estimator': base_estimator,
            'n_estimators': 10,
            'max_samples': 1.0,
            'max_features': 1.0,
            'bootstrap': True,
            'bootstrap_features': False,
            'n_jobs': -1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__("bagging_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> BaggingClassifier:
        """Create Bagging Classifier."""
        return BaggingClassifier(**kwargs)


class StackingRegressorModel(EnsembleModel):
    """Stacking regressor ensemble model wrapper."""
    
    def __init__(self, base_models: List[Tuple[str, Any]], final_estimator: Any = None, **kwargs):
        """
        Initialize stacking regressor.
        
        Args:
            base_models: List of (name, model) tuples
            final_estimator: Final estimator for stacking
            **kwargs: Additional parameters
        """
        self.base_models = base_models
        self.final_estimator = final_estimator
        
        default_params = {
            'estimators': base_models,
            'final_estimator': final_estimator,
            'cv': 5,
            'n_jobs': -1,
            'stack_method': 'predict',
            'passthrough': False
        }
        default_params.update(kwargs)
        
        super().__init__("stacking_regressor", "regression", **default_params)
    
    def _create_model(self, **kwargs) -> StackingRegressor:
        """Create Stacking Regressor."""
        return StackingRegressor(**kwargs)
    
    def get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get meta-features from base models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of meta-features
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting meta-features")
        
        if not hasattr(self.model, 'transform'):
            raise ValueError("Model does not support transform method")
        
        X_processed = self.preprocess_features(X)
        return self.model.transform(X_processed)


class StackingClassifierModel(EnsembleModel):
    """Stacking classifier ensemble model wrapper."""
    
    def __init__(self, base_models: List[Tuple[str, Any]], final_estimator: Any = None, **kwargs):
        """
        Initialize stacking classifier.
        
        Args:
            base_models: List of (name, model) tuples
            final_estimator: Final estimator for stacking
            **kwargs: Additional parameters
        """
        self.base_models = base_models
        self.final_estimator = final_estimator
        
        default_params = {
            'estimators': base_models,
            'final_estimator': final_estimator,
            'cv': 5,
            'n_jobs': -1,
            'stack_method': 'predict_proba',
            'passthrough': False
        }
        default_params.update(kwargs)
        
        super().__init__("stacking_classifier", "classification", **default_params)
    
    def _create_model(self, **kwargs) -> StackingClassifier:
        """Create Stacking Classifier."""
        return StackingClassifier(**kwargs)
    
    def get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get meta-features from base models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of meta-features
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting meta-features")
        
        if not hasattr(self.model, 'transform'):
            raise ValueError("Model does not support transform method")
        
        X_processed = self.preprocess_features(X)
        return self.model.transform(X_processed)


class CustomEnsembleModel(EnsembleModel):
    """Custom ensemble model wrapper."""
    
    def __init__(self, base_models: List[Any], ensemble_method: str = 'average', **kwargs):
        """
        Initialize custom ensemble.
        
        Args:
            base_models: List of base models
            ensemble_method: Method for combining predictions ('average', 'weighted', 'max')
            **kwargs: Additional parameters
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        
        super().__init__("custom_ensemble", "regression", **kwargs)
        self.model = None  # Custom ensemble doesn't use sklearn model
    
    def _create_model(self, **kwargs) -> None:
        """Custom ensemble doesn't create sklearn model."""
        return None
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'CustomEnsembleModel':
        """
        Fit the custom ensemble model.
        
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
        
        # Fit base models
        try:
            for model in self.base_models:
                if hasattr(model, 'fit'):
                    model.fit(X, y, **kwargs)
            
            self.is_fitted = True
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self._calculate_training_metrics(y, y_pred)
            
            # Calculate validation metrics if provided
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                self._calculate_validation_metrics(y_val, y_val_pred)
            
            logger.info(f"Custom ensemble model fitted successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit custom ensemble: {e}")
            raise
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the custom ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Get predictions from base models
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                base_predictions.append(pred)
        
        if not base_predictions:
            raise ValueError("No valid predictions from base models")
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            return np.mean(base_predictions, axis=0)
        elif self.ensemble_method == 'weighted':
            # Equal weights for now - could be extended
            weights = np.ones(len(base_predictions)) / len(base_predictions)
            return np.average(base_predictions, axis=0, weights=weights)
        elif self.ensemble_method == 'max':
            return np.max(base_predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting base predictions")
        
        base_predictions = {}
        
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                base_predictions[f'base_model_{i}'] = pred
        
        return base_predictions
