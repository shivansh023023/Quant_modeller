"""
Utility functions for feature engineering and management.

This module provides helper functions for feature preprocessing,
validation, and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureUtils:
    """
    Utility functions for feature engineering and management.
    
    This class provides methods for feature preprocessing, selection,
    analysis, and quality assessment.
    """
    
    def __init__(self):
        """Initialize the feature utilities."""
        self.scalers = {}
        self.pca_models = {}
    
    def preprocess_features(
        self, 
        features: pd.DataFrame,
        method: str = "standard",
        handle_missing: str = "forward_fill",
        handle_outliers: str = "winsorize",
        winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    ) -> pd.DataFrame:
        """
        Preprocess features for machine learning.
        
        Args:
            features: DataFrame of features
            method: Scaling method ('standard', 'robust', 'minmax', 'none')
            handle_missing: Method to handle missing values
            handle_outliers: Method to handle outliers
            winsorize_limits: Limits for winsorization
            
        Returns:
            Preprocessed features DataFrame
        """
        processed_features = features.copy()
        
        # Handle missing values
        if handle_missing == "forward_fill":
            processed_features = processed_features.fillna(method='ffill').fillna(method='bfill')
        elif handle_missing == "interpolate":
            processed_features = processed_features.interpolate(method='linear')
        elif handle_missing == "drop":
            processed_features = processed_features.dropna()
        elif handle_missing == "zero":
            processed_features = processed_features.fillna(0)
        
        # Handle outliers
        if handle_outliers == "winsorize":
            processed_features = self._winsorize_features(processed_features, winsorize_limits)
        elif handle_outliers == "clip":
            processed_features = self._clip_features(processed_features)
        elif handle_outliers == "zscore":
            processed_features = self._remove_outliers_zscore(processed_features)
        
        # Scale features
        if method != "none":
            processed_features = self._scale_features(processed_features, method)
        
        return processed_features
    
    def _winsorize_features(
        self, 
        features: pd.DataFrame, 
        limits: Tuple[float, float]
    ) -> pd.DataFrame:
        """Winsorize features to handle outliers."""
        winsorized = features.copy()
        
        for col in winsorized.columns:
            if winsorized[col].dtype in ['float64', 'int64']:
                winsorized[col] = stats.mstats.winsorize(
                    winsorized[col], limits=limits
                )
        
        return winsorized
    
    def _clip_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clip features to handle outliers."""
        clipped = features.copy()
        
        for col in clipped.columns:
            if clipped[col].dtype in ['float64', 'int64']:
                q1 = clipped[col].quantile(0.01)
                q99 = clipped[col].quantile(0.99)
                clipped[col] = clipped[col].clip(lower=q1, upper=q99)
        
        return clipped
    
    def _remove_outliers_zscore(self, features: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        cleaned = features.copy()
        
        for col in cleaned.columns:
            if cleaned[col].dtype in ['float64', 'int64']:
                z_scores = np.abs(stats.zscore(cleaned[col].dropna()))
                mask = z_scores < threshold
                cleaned.loc[~mask, col] = np.nan
        
        return cleaned
    
    def _scale_features(self, features: pd.DataFrame, method: str) -> pd.DataFrame:
        """Scale features using specified method."""
        scaled = features.copy()
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            return scaled
        
        # Fit scaler and transform
        scaled_values = scaler.fit_transform(scaled)
        scaled = pd.DataFrame(scaled_values, index=scaled.index, columns=scaled.columns)
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        return scaled
    
    def select_features(
        self, 
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "mutual_info",
        k: int = 50,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features.
        
        Args:
            features: Feature DataFrame
            target: Target variable
            method: Selection method ('mutual_info', 'f_regression', 'correlation', 'variance')
            k: Number of features to select
            threshold: Threshold for selection methods
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if method == "mutual_info":
            return self._select_features_mutual_info(features, target, k)
        elif method == "f_regression":
            return self._select_features_f_regression(features, target, k)
        elif method == "correlation":
            return self._select_features_correlation(features, target, threshold)
        elif method == "variance":
            return self._select_features_variance(features, threshold)
        else:
            logger.warning(f"Unknown selection method: {method}")
            return features, list(features.columns)
    
    def _select_features_mutual_info(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        k: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using mutual information."""
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        target_clean = target.fillna(method='ffill').fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(features_clean, target_clean, random_state=42)
        
        # Select top k features
        feature_scores = list(zip(features.columns, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [name for name, score in feature_scores[:k]]
        
        return features[selected_features], selected_features
    
    def _select_features_f_regression(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        k: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using F-regression."""
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        target_clean = target.fillna(method='ffill').fillna(0)
        
        # Calculate F-scores
        f_scores, _ = f_regression(features_clean, target_clean)
        
        # Select top k features
        feature_scores = list(zip(features.columns, f_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [name for name, score in feature_scores[:k]]
        
        return features[selected_features], selected_features
    
    def _select_features_correlation(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        threshold: float
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using correlation with target."""
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        
        # Select features above threshold
        selected_features = correlations[correlations > threshold].index.tolist()
        
        if not selected_features:
            logger.warning("No features meet correlation threshold, selecting top 10")
            selected_features = correlations.nlargest(10).index.tolist()
        
        return features[selected_features], selected_features
    
    def _select_features_variance(
        self, 
        features: pd.DataFrame, 
        threshold: float
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using variance threshold."""
        # Calculate variances
        variances = features.var()
        
        # Select features above threshold
        selected_features = variances[variances > threshold].index.tolist()
        
        if not selected_features:
            logger.warning("No features meet variance threshold, selecting top 10")
            selected_features = variances.nlargest(10).index.tolist()
        
        return features[selected_features], selected_features
    
    def reduce_dimensions(
        self, 
        features: pd.DataFrame,
        method: str = "pca",
        n_components: Union[int, float] = 0.95
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reduce feature dimensions using dimensionality reduction.
        
        Args:
            features: Feature DataFrame
            method: Reduction method ('pca', 'feature_agglomeration')
            n_components: Number of components or explained variance ratio
            
        Returns:
            Tuple of (reduced_features, reduction_info)
        """
        if method == "pca":
            return self._reduce_dimensions_pca(features, n_components)
        else:
            logger.warning(f"Unknown reduction method: {method}")
            return features, {}
    
    def _reduce_dimensions_pca(
        self, 
        features: pd.DataFrame, 
        n_components: Union[int, float]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Reduce dimensions using PCA."""
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced_features = pca.fit_transform(features_clean)
        
        # Create DataFrame
        if isinstance(n_components, float):
            n_components = reduced_features.shape[1]
        
        column_names = [f"PC_{i+1}" for i in range(n_components)]
        reduced_df = pd.DataFrame(
            reduced_features, 
            index=features_clean.index, 
            columns=column_names
        )
        
        # Store PCA model
        self.pca_models['pca'] = pca
        
        # Create info dictionary
        info = {
            'method': 'pca',
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'feature_importance': dict(zip(features.columns, pca.components_[0]))
        }
        
        return reduced_df, info
    
    def analyze_feature_quality(
        self, 
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze the quality of features.
        
        Args:
            features: Feature DataFrame
            target: Target variable (optional)
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['basic_stats'] = {
            'n_features': len(features.columns),
            'n_observations': len(features),
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Missing values
        missing_info = features.isnull().sum()
        quality_metrics['missing_values'] = {
            'total_missing': missing_info.sum(),
            'missing_percentage': (missing_info.sum() / (len(features) * len(features.columns))) * 100,
            'features_with_missing': missing_info[missing_info > 0].to_dict()
        }
        
        # Data types
        quality_metrics['data_types'] = features.dtypes.value_counts().to_dict()
        
        # Feature correlations
        if len(features.columns) > 1:
            corr_matrix = features.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.95:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            quality_metrics['high_correlations'] = high_corr_pairs
        
        # Feature variance
        variances = features.var()
        quality_metrics['variance'] = {
            'low_variance_features': variances[variances < 1e-6].index.tolist(),
            'variance_statistics': {
                'min': variances.min(),
                'max': variances.max(),
                'mean': variances.mean(),
                'median': variances.median()
            }
        }
        
        # Target relationship (if provided)
        if target is not None:
            quality_metrics['target_relationship'] = self._analyze_target_relationship(features, target)
        
        return quality_metrics
    
    def _analyze_target_relationship(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Dict[str, Any]:
        """Analyze relationship between features and target."""
        relationship_info = {}
        
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        relationship_info['correlations'] = {
            'mean_correlation': correlations.mean(),
            'max_correlation': correlations.max(),
            'top_correlated_features': correlations.nlargest(10).to_dict()
        }
        
        # Calculate mutual information (sample for large datasets)
        if len(features) > 10000:
            sample_size = 10000
            sample_idx = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features.iloc[sample_idx]
            target_sample = target.iloc[sample_idx]
        else:
            features_sample = features
            target_sample = target
        
        # Handle missing values for mutual information
        features_clean = features_sample.fillna(method='ffill').fillna(0)
        target_clean = target_sample.fillna(method='ffill').fillna(0)
        
        try:
            mi_scores = mutual_info_regression(features_clean, target_clean, random_state=42)
            relationship_info['mutual_information'] = {
                'mean_mi': mi_scores.mean(),
                'max_mi': mi_scores.max(),
                'top_mi_features': dict(zip(features.columns, mi_scores))
            }
        except Exception as e:
            logger.warning(f"Could not calculate mutual information: {e}")
            relationship_info['mutual_information'] = None
        
        return relationship_info
    
    def create_feature_groups(
        self, 
        features: pd.DataFrame,
        grouping_method: str = "correlation"
    ) -> Dict[str, List[str]]:
        """
        Group features based on similarity.
        
        Args:
            features: Feature DataFrame
            grouping_method: Method for grouping ('correlation', 'clustering')
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        if grouping_method == "correlation":
            return self._group_features_correlation(features)
        else:
            logger.warning(f"Unknown grouping method: {grouping_method}")
            return {"group_1": list(features.columns)}
    
    def _group_features_correlation(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features based on correlation."""
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Create feature groups
        groups = {}
        used_features = set()
        group_id = 1
        
        for feature in features.columns:
            if feature in used_features:
                continue
            
            # Find highly correlated features
            correlated_features = [feature]
            used_features.add(feature)
            
            for other_feature in features.columns:
                if other_feature not in used_features:
                    if corr_matrix.loc[feature, other_feature] > 0.8:
                        correlated_features.append(other_feature)
                        used_features.add(other_feature)
            
            groups[f"group_{group_id}"] = correlated_features
            group_id += 1
        
        return groups
    
    def get_feature_importance_ranking(
        self, 
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "mutual_info"
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Args:
            features: Feature DataFrame
            target: Target variable
            method: Importance method ('mutual_info', 'correlation', 'variance')
            
        Returns:
            DataFrame with feature importance rankings
        """
        if method == "mutual_info":
            return self._get_mutual_info_ranking(features, target)
        elif method == "correlation":
            return self._get_correlation_ranking(features, target)
        elif method == "variance":
            return self._get_variance_ranking(features)
        else:
            logger.warning(f"Unknown importance method: {method}")
            return pd.DataFrame()
    
    def _get_mutual_info_ranking(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> pd.DataFrame:
        """Get feature importance ranking using mutual information."""
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        target_clean = target.fillna(method='ffill').fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(features_clean, target_clean, random_state=42)
        
        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'feature': features.columns,
            'mutual_info': mi_scores,
            'rank': np.argsort(mi_scores)[::-1] + 1
        })
        
        return ranking.sort_values('rank')
    
    def _get_correlation_ranking(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> pd.DataFrame:
        """Get feature importance ranking using correlation."""
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        
        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'rank': np.argsort(correlations.values)[::-1] + 1
        })
        
        return ranking.sort_values('rank')
    
    def _get_variance_ranking(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance ranking using variance."""
        # Calculate variances
        variances = features.var()
        
        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'feature': variances.index,
            'variance': variances.values,
            'rank': np.argsort(variances.values)[::-1] + 1
        })
        
        return ranking.sort_values('rank')
    
    def validate_feature_expression(
        self, 
        expression: str,
        available_features: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a feature expression.
        
        Args:
            expression: Feature expression string
            available_features: List of available feature names
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for basic syntax
        try:
            # Simple validation - check if expression can be evaluated
            # This is a basic check and could be enhanced
            if any(char in expression for char in ['import', 'exec', 'eval']):
                errors.append("Expression contains forbidden operations")
        except Exception as e:
            errors.append(f"Expression syntax error: {e}")
        
        # Check for feature references
        for feature in available_features:
            if feature in expression:
                # Check if feature name is properly referenced
                if not any(op in expression for op in ['+', '-', '*', '/', '(', ')', ' ']):
                    errors.append(f"Feature {feature} may not be properly referenced")
        
        # Check for potential data leakage
        if any(term in expression.lower() for term in ['future', 'tomorrow', 'next', 'lead']):
            errors.append("Expression may contain future information")
        
        return len(errors) == 0, errors
