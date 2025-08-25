"""
Scikit-learn based models for quantitative trading strategies.

This module provides implementations of various scikit-learn models
wrapped in the BaseModel interface for consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """
    Wrapper for scikit-learn models.
    
    This class provides a consistent interface for various scikit-learn
    models while maintaining the BaseModel contract.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the sklearn model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('regression' or 'classification')
            **kwargs: Model parameters
        """
        super().__init__(model_name, model_type, **kwargs)
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> Any:
        """
        Create the underlying sklearn model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Sklearn model instance
        """
        model_name = self.model_name.lower()
        
        if self.model_type == 'regression':
            if 'linear' in model_name:
                return LinearRegression(**kwargs)
            elif 'ridge' in model_name:
                return Ridge(**kwargs)
            elif 'lasso' in model_name:
                return Lasso(**kwargs)
            elif 'elastic' in model_name:
                return ElasticNet(**kwargs)
            elif 'sgd' in model_name:
                return SGDRegressor(**kwargs)
            elif 'svr' in model_name or 'svm' in model_name:
                return SVR(**kwargs)
            elif 'knn' in model_name or 'neighbors' in model_name:
                return KNeighborsRegressor(**kwargs)
            elif 'mlp' in model_name or 'neural' in model_name:
                return MLPRegressor(**kwargs)
            else:
                # Default to linear regression
                return LinearRegression(**kwargs)
        
        elif self.model_type == 'classification':
            if 'logistic' in model_name:
                return LogisticRegression(**kwargs)
            elif 'sgd' in model_name:
                return SGDClassifier(**kwargs)
            elif 'svc' in model_name or 'svm' in model_name:
                return SVC(**kwargs)
            elif 'knn' in model_name or 'neighbors' in model_name:
                return KNeighborsClassifier(**kwargs)
            elif 'mlp' in model_name or 'neural' in model_name:
                return MLPClassifier(**kwargs)
            else:
                # Default to logistic regression
                return LogisticRegression(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'SklearnModel':
        """
        Fit the sklearn model to the training data.
        
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
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Store in both training_history and training_metrics for compatibility
            self.training_history.update(metrics)
            self.training_metrics.update(metrics)
        else:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Store in both training_history and training_metrics for compatibility
            self.training_history.update(metrics)
            self.training_metrics.update(metrics)
    
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
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        else:
            logger.warning("Model does not support feature importance")
            return {}
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return self.model_params
    
    def set_model_params(self, **params) -> None:
        """
        Set new model parameters.
        
        Args:
            **params: New model parameters
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
            self.model_params.update(params)
        else:
            logger.warning("Model does not support set_params")
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_splits: int = 5,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv_splits: Number of cross-validation splits
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        if not self.is_fitted:
            # Fit the model first
            self.fit(X, y, **kwargs)
        
        X_processed = self.preprocess_features(X)
        
        # Define scoring metrics based on model type
        if self.model_type == 'regression':
            scoring_metrics = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        else:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    self.model, X_processed, y, 
                    cv=cv_splits, scoring=metric
                )
                cv_results[metric] = scores.tolist()
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                cv_results[metric] = []
        
        return cv_results


class LinearRegressionModel(SklearnModel):
    """Linear regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("linear_regression", "regression", **kwargs)


class LogisticRegressionModel(SklearnModel):
    """Logistic regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("logistic_regression", "classification", **kwargs)


class RidgeRegressionModel(SklearnModel):
    """Ridge regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("ridge_regression", "regression", **kwargs)


class LassoRegressionModel(SklearnModel):
    """Lasso regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("lasso_regression", "regression", **kwargs)


class ElasticNetModel(SklearnModel):
    """Elastic net model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("elastic_net", "regression", **kwargs)


class SVRModel(SklearnModel):
    """Support Vector Regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("svr", "regression", **kwargs)


class SVCModel(SklearnModel):
    """Support Vector Classification model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("svc", "classification", **kwargs)


class KNNRegressorModel(SklearnModel):
    """K-Nearest Neighbors regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("knn_regressor", "regression", **kwargs)


class KNNClassifierModel(SklearnModel):
    """K-Nearest Neighbors classification model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("knn_classifier", "classification", **kwargs)


class MLPRegressorModel(SklearnModel):
    """Multi-layer Perceptron regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("mlp_regressor", "regression", **kwargs)


class MLPClassifierModel(SklearnModel):
    """Multi-layer Perceptron classification model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("mlp_classifier", "classification", **kwargs)


class SGDRegressorModel(SklearnModel):
    """Stochastic Gradient Descent regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("sgd_regressor", "regression", **kwargs)


class SGDClassifierModel(SklearnModel):
    """Stochastic Gradient Descent classification model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__("sgd_classifier", "classification", **kwargs)
