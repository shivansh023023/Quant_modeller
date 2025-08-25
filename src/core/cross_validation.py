"""
Cross-validation strategies for time series data.

This module provides specialized cross-validation methods that
prevent data leakage in time series contexts, including walk-forward
and purged K-fold validation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
import warnings


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-forward cross-validation for time series data.
    
    This method creates expanding training windows with fixed-size
    test windows, ensuring no future information leaks into training.
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        test_size: int = 63,  # ~3 months
        min_train_size: int = 252,  # ~1 year
        step_size: Optional[int] = None
    ):
        """
        Initialize walk-forward cross-validation.
        
        Args:
            n_splits: Number of CV splits
            test_size: Size of test set in periods
            min_train_size: Minimum training set size in periods
            step_size: Step size between splits (defaults to test_size)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.step_size = step_size if step_size is not None else test_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features array
            y: Target array
            groups: Group labels (ignored)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if n_samples < self.min_train_size + self.test_size:
            raise ValueError(
                f"Data too small for walk-forward CV. "
                f"Need at least {self.min_train_size + self.test_size} samples, "
                f"got {n_samples}."
            )
        
        # Calculate start positions for each split
        start_positions = []
        current_pos = self.min_train_size
        
        for _ in range(self.n_splits):
            if current_pos + self.test_size > n_samples:
                break
            
            start_positions.append(current_pos)
            current_pos += self.step_size
        
        # Adjust n_splits if we can't make enough splits
        actual_splits = len(start_positions)
        if actual_splits < self.n_splits:
            warnings.warn(
                f"Requested {self.n_splits} splits but only {actual_splits} "
                f"are possible with the given data size and parameters."
            )
        
        # Generate splits
        for start_pos in start_positions:
            train_end = start_pos
            test_end = min(start_pos + self.test_size, n_samples)
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(start_pos, test_end)
            
            yield train_indices, test_indices


class PurgedKFoldCV(BaseCrossValidator):
    """
    Purged K-fold cross-validation for time series data.
    
    This method removes overlapping periods between training and test sets
    to prevent data leakage, with optional embargo periods.
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        purge_period: int = 10,
        embargo_period: int = 5
    ):
        """
        Initialize purged K-fold cross-validation.
        
        Args:
            n_splits: Number of CV splits
            purge_period: Number of periods to purge around test set
            embargo_period: Number of periods embargo after test set
        """
        self.n_splits = n_splits
        self.purge_period = purge_period
        self.embargo_period = embargo_period
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features array
            y: Target array
            groups: Group labels (ignored)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if n_samples < self.n_splits * (self.purge_period + self.embargo_period):
            raise ValueError(
                f"Data too small for purged K-fold CV. "
                f"Need at least {self.n_splits * (self.purge_period + self.embargo_period)} samples, "
                f"got {n_samples}."
            )
        
        # Calculate split sizes
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = i * split_size
            test_end = test_start + split_size
            
            # Calculate purge boundaries
            purge_start = max(0, test_start - self.purge_period)
            purge_end = min(n_samples, test_end + self.embargo_period)
            
            # Create training indices (excluding purged periods)
            train_indices = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n_samples)
            ])
            
            # Create test indices
            test_indices = np.arange(test_start, test_end)
            
            # Ensure we have enough training data
            if len(train_indices) < split_size:
                continue
            
            yield train_indices, test_indices


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time series split cross-validation.
    
    Simple time-based split that respects temporal order.
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        """
        Initialize time series split.
        
        Args:
            n_splits: Number of CV splits
            test_size: Proportion of data for test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features array
            y: Target array
            groups: Group labels (ignored)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        
        # Calculate the step size to ensure we get exactly n_splits
        step_size = (n_samples - test_samples) // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            train_end = step_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + test_samples, n_samples)
            
            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices


class CrossValidationManager:
    """
    Manager for cross-validation strategies.
    
    This class provides a unified interface for different CV methods
    and handles the creation of training/validation splits.
    """
    
    def __init__(self, cv_method: str = "walk_forward", **cv_params):
        """
        Initialize CV manager.
        
        Args:
            cv_method: CV method ('walk_forward', 'purged_kfold', 'time_series')
            **cv_params: Parameters for the CV method
        """
        self.cv_method = cv_method
        self.cv_params = cv_params
        self.cv_object = self._create_cv_object()
    
    def _create_cv_object(self):
        """Create the appropriate CV object."""
        if self.cv_method == "walk_forward":
            return WalkForwardCV(**self.cv_params)
        elif self.cv_method == "purged_kfold":
            return PurgedKFoldCV(**self.cv_params)
        elif self.cv_method == "time_series":
            return TimeSeriesSplit(**self.cv_params)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")
    
    def get_splits(self, X, y=None):
        """Get CV splits."""
        return self.cv_object.split(X, y)
    
    def get_n_splits(self):
        """Get number of splits."""
        return self.cv_object.get_n_splits()
    
    def validate_split(self, train_indices: np.ndarray, test_indices: np.ndarray) -> bool:
        """
        Validate that a split doesn't have data leakage.
        
        Args:
            train_indices: Training set indices
            test_indices: Test set indices
            
        Returns:
            True if split is valid, False otherwise
        """
        # Check for overlap
        if len(np.intersect1d(train_indices, test_indices)) > 0:
            return False
        
        # Check temporal order (for time series)
        if len(train_indices) > 0 and len(test_indices) > 0:
            max_train_idx = np.max(train_indices)
            min_test_idx = np.min(test_indices)
            
            # For time series, test should come after training
            if min_test_idx <= max_train_idx:
                return False
        
        return True
    
    def get_split_info(self, X, y=None) -> Dict[str, Any]:
        """
        Get information about CV splits.
        
        Args:
            X: Features array
            y: Target array
            
        Returns:
            Dictionary with split information
        """
        splits = list(self.get_splits(X, y))
        
        split_info = {
            "method": self.cv_method,
            "n_splits": len(splits),
            "n_samples": len(X),
            "splits": []
        }
        
        for i, (train_idx, test_idx) in enumerate(splits):
            split_data = {
                "split": i,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start": train_idx[0] if len(train_idx) > 0 else None,
                "train_end": train_idx[-1] if len(train_idx) > 0 else None,
                "test_start": test_idx[0] if len(test_idx) > 0 else None,
                "test_end": test_idx[-1] if len(test_idx) > 0 else None,
                "is_valid": self.validate_split(train_idx, test_idx)
            }
            split_info["splits"].append(split_data)
        
        return split_info
    
    def fit_and_validate(
        self, 
        estimator: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray,
        scoring: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Fit estimator on all CV folds and return validation scores.
        
        Args:
            estimator: Scikit-learn compatible estimator
            X: Features array
            y: Target array
            scoring: Scoring function (optional)
            
        Returns:
            Dictionary with validation results
        """
        scores = []
        split_results = []
        
        for i, (train_idx, test_idx) in enumerate(self.get_splits(X, y)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit estimator
            estimator.fit(X_train, y_train)
            
            # Predict
            y_pred = estimator.predict(X_test)
            
            # Calculate score
            if scoring is not None:
                score = scoring(y_test, y_pred)
            else:
                # Default to R² for regression, accuracy for classification
                if hasattr(estimator, 'score'):
                    score = estimator.score(X_test, y_test)
                else:
                    score = np.nan
            
            scores.append(score)
            
            # Store split results
            split_result = {
                "split": i,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "score": score,
                "train_indices": train_idx,
                "test_indices": test_idx
            }
            split_results.append(split_result)
        
        # Calculate summary statistics
        results = {
            "scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "split_results": split_results,
            "cv_method": self.cv_method,
            "n_splits": len(scores)
        }
        
        return results


def create_cv_manager(
    cv_method: str = "walk_forward",
    **cv_params
) -> CrossValidationManager:
    """
    Factory function to create CV manager.
    
    Args:
        cv_method: CV method name
        **cv_params: CV parameters
        
    Returns:
        CrossValidationManager instance
    """
    return CrossValidationManager(cv_method, **cv_params)


def validate_cv_split(
    train_indices: np.ndarray, 
    test_indices: np.ndarray,
    max_lookback: int = 0
) -> Tuple[bool, List[str]]:
    """
    Validate CV split for potential data leakage.
    
    Args:
        train_indices: Training set indices
        test_indices: Test set indices
        max_lookback: Maximum lookback period for features
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for overlap
    if len(np.intersect1d(train_indices, test_indices)) > 0:
        errors.append("Training and test sets overlap")
    
    # Check temporal order
    if len(train_indices) > 0 and len(test_indices) > 0:
        max_train_idx = np.max(train_indices)
        min_test_idx = np.min(test_indices)
        
        if min_test_idx <= max_train_idx:
            errors.append("Test set contains data from before training set")
    
    # Check lookback requirements
    if max_lookback > 0:
        if len(train_indices) < max_lookback:
            errors.append(f"Training set too small for max lookback {max_lookback}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors
