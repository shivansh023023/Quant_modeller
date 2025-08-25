"""
Base model class for all machine learning models.

This module defines the base interface that all models must implement
for consistency across the quant-lab system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the interface that all models must implement,
    ensuring consistency across different model types and enabling
    easy model switching and comparison.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'classification', 'regression')
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        self.training_history = {}
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Model metadata
        self.metadata = {
            'created_at': pd.Timestamp.now(),
            'model_name': model_name,
            'model_type': model_type,
            'model_params': kwargs,
            'version': '1.0.0'
        }
    
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional validation data tuple (X_val, y_val)
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        pass
    
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
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_params': self.model_params,
            'metadata': self.metadata
        }
        
        if self.is_fitted:
            summary.update({
                'training_history': self.training_history,
                'validation_metrics': self.validation_metrics
            })
        
        return summary
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model object
            if hasattr(self.model, 'save_model'):  # For models like LightGBM
                self.model.save_model(filepath)
            else:
                joblib.dump(self.model, filepath)
            
            # Save metadata separately
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'feature_names': self.feature_names,
                    'target_name': self.target_name,
                    'training_history': self.training_history,
                    'validation_metrics': self.validation_metrics,
                    'metadata': self.metadata
                }, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            # Load model object
            if filepath.endswith('.txt'):  # LightGBM model
                # This would need to be implemented based on the specific model type
                logger.warning("LightGBM model loading not implemented yet")
                return False
            else:
                self.model = joblib.load(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            if Path(metadata_path).exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.feature_names = metadata.get('feature_names')
                    self.target_name = metadata.get('target_name')
                    self.training_history = metadata.get('training_history', {})
                    self.validation_metrics = metadata.get('validation_metrics', {})
                    self.metadata.update(metadata.get('metadata', {}))
            
            self.is_fitted = True
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update model metadata.
        
        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = pd.Timestamp.now()
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the training history of the model.
        
        Returns:
            Dictionary containing training history
        """
        return self.training_history.copy()
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """
        Get validation metrics from training.
        
        Returns:
            Dictionary containing validation metrics
        """
        return self.validation_metrics.copy()
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set the feature names for the model.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def set_target_name(self, target_name: str) -> None:
        """
        Set the target name for the model.
        
        Args:
            target_name: Name of the target variable
        """
        self.target_name = target_name
    
    def validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """
        Validate input data for the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                logger.error("X must be a pandas DataFrame")
                return False
            
            # Check if X has data
            if X.empty:
                logger.error("X is empty")
                return False
            
            # Check if y is provided and valid
            if y is not None:
                if not isinstance(y, pd.Series):
                    logger.error("y must be a pandas Series")
                    return False
                
                if y.empty:
                    logger.error("y is empty")
                    return False
                
                if len(X) != len(y):
                    logger.error("X and y must have the same length")
                    return False
            
            # Check for missing values
            if X.isnull().any().any():
                logger.warning("X contains missing values")
            
            if y is not None and y.isnull().any():
                logger.warning("y contains missing values")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Preprocessed feature DataFrame
        """
        # Default preprocessing - can be overridden by subclasses
        processed_X = X.copy()
        
        # Handle missing values
        if processed_X.isnull().any().any():
            processed_X = processed_X.fillna(method='ffill').fillna(0)
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(processed_X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    processed_X[feature] = 0
            
            # Reorder columns to match training
            processed_X = processed_X[self.feature_names]
        
        return processed_X
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Get information about the model size.
        
        Returns:
            Dictionary containing size information
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        try:
            # Estimate model size
            if hasattr(self.model, 'n_features_in_'):
                n_features = self.model.n_features_in_
            elif self.feature_names:
                n_features = len(self.feature_names)
            else:
                n_features = 'unknown'
            
            # Get model parameters count
            if hasattr(self.model, 'n_parameters'):
                n_params = self.model.n_parameters
            else:
                n_params = 'unknown'
            
            return {
                'n_features': n_features,
                'n_parameters': n_params,
                'model_object_size': str(type(self.model))
            }
            
        except Exception as e:
            logger.warning(f"Could not determine model size: {e}")
            return {'error': str(e)}
    
    def clone(self) -> 'BaseModel':
        """
        Create a copy of the model.
        
        Returns:
            Copy of the model
        """
        # Create new instance with same parameters
        new_model = self.__class__(self.model_name, self.model_type, **self.model_params)
        
        # Copy metadata
        new_model.metadata = self.metadata.copy()
        new_model.feature_names = self.feature_names.copy() if self.feature_names else None
        new_model.target_name = self.target_name
        
        return new_model
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', fitted={self.is_fitted}, params={self.model_params})"
