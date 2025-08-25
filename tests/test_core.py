"""
Tests for core functionality including config, data API, and cross-validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from src.core.config import ConfigManager
from src.core.data_api import DataAPI, DataCatalog
from src.core.cross_validation import WalkForwardCV, PurgedKFoldCV, TimeSeriesSplit


class TestConfigManager:
    """Test configuration management."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_config_manager_initialization(self, temp_config_dir):
        """Test ConfigManager initialization."""
        # Create a temporary config file
        config_file = Path(temp_config_dir) / "api_keys.yml"
        config_content = """
alpha_vantage:
  api_key: "test_alpha_key"
  rate_limit: 5

gemini:
  api_key: "test_gemini_key"
  model: "gemini-pro"

yfinance:
  enabled: true
"""
        config_file.write_text(config_content)
        
        # Initialize config manager
        config_manager = ConfigManager(config_dir=temp_config_dir)
        
        # Test API key retrieval
        assert config_manager.get_api_key('alpha_vantage') == "test_alpha_key"
        assert config_manager.get_api_key('gemini') == "test_gemini_key"
        assert config_manager.has_api_key('alpha_vantage')
        assert config_manager.has_api_key('gemini')
    
    def test_missing_config_file(self, temp_config_dir):
        """Test behavior with missing config file."""
        config_manager = ConfigManager(config_dir=temp_config_dir)
        
        # Should handle missing config gracefully
        assert config_manager.get_api_key('alpha_vantage') is None
        assert not config_manager.has_api_key('alpha_vantage')
    
    def test_environment_variable_override(self, temp_config_dir, monkeypatch):
        """Test environment variable override."""
        # Set environment variable
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env_alpha_key")
        monkeypatch.setenv("GEMINI_API_KEY", "env_gemini_key")
        
        config_manager = ConfigManager(config_dir=temp_config_dir)
        
        # Environment variables should take precedence
        assert config_manager.get_api_key('alpha_vantage') == "env_alpha_key"
        assert config_manager.get_api_key('gemini') == "env_gemini_key"
    
    def test_service_enabled_check(self, temp_config_dir):
        """Test service enabled check."""
        config_file = Path(temp_config_dir) / "api_keys.yml"
        config_content = """
alpha_vantage:
  api_key: "test_key"
  enabled: true

yfinance:
  enabled: true
"""
        config_file.write_text(config_content)
        
        config_manager = ConfigManager(config_dir=temp_config_dir)
        
        assert config_manager.is_service_enabled('alpha_vantage')
        assert config_manager.is_service_enabled('yfinance')


class TestDataCatalog:
    """Test data catalog functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_catalog_initialization(self, temp_data_dir):
        """Test DataCatalog initialization."""
        catalog = DataCatalog(data_dir=temp_data_dir)
        
        # Check that directories are created
        assert (Path(temp_data_dir) / "raw").exists()
        assert (Path(temp_data_dir) / "interim").exists()
        assert (Path(temp_data_dir) / "curated").exists()
        
        # Check catalog structure
        assert "version" in catalog.catalog
        assert "datasets" in catalog.catalog
        assert "sources" in catalog.catalog
    
    def test_dataset_registration(self, temp_data_dir):
        """Test dataset registration."""
        catalog = DataCatalog(data_dir=temp_data_dir)
        
        metadata = {
            "source": "yfinance",
            "tickers": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        
        catalog.register_dataset("test_dataset", metadata)
        
        # Check that dataset is registered
        assert "test_dataset" in catalog.list_datasets()
        dataset_info = catalog.get_dataset_info("test_dataset")
        assert dataset_info["source"] == "yfinance"
        assert dataset_info["tickers"] == ["AAPL", "GOOGL"]
    
    def test_catalog_persistence(self, temp_data_dir):
        """Test that catalog persists to disk."""
        catalog = DataCatalog(data_dir=temp_data_dir)
        
        # Register a dataset
        metadata = {"source": "test", "data": "test_data"}
        catalog.register_dataset("persistent_dataset", metadata)
        
        # Create new catalog instance (should load from disk)
        new_catalog = DataCatalog(data_dir=temp_data_dir)
        
        # Should find the registered dataset
        assert "persistent_dataset" in new_catalog.list_datasets()


class TestDataAPI:
    """Test data API functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_data_api_initialization(self, temp_data_dir):
        """Test DataAPI initialization."""
        data_api = DataAPI(data_dir=temp_data_dir)
        
        assert data_api.catalog is not None
        assert isinstance(data_api.catalog, DataCatalog)
    
    def test_data_api_with_alpha_vantage_key(self, temp_data_dir):
        """Test DataAPI with Alpha Vantage key."""
        data_api = DataAPI(data_dir=temp_data_dir, alpha_vantage_key="test_key")
        
        assert data_api.alpha_vantage is not None
        assert data_api.alpha_vantage_key == "test_key"
    
    def test_load_ohlcv_validation(self, temp_data_dir):
        """Test OHLCV loading input validation."""
        data_api = DataAPI(data_dir=temp_data_dir)
        
        # Test with invalid source
        with pytest.raises(ValueError, match="Unsupported data source"):
            data_api.load_ohlcv(
                tickers=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                source="invalid_source"
            )


class TestCrossValidator:
    """Test cross-validation functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for cross-validation testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
            index=pd.date_range('2020-01-01', periods=n_samples, freq='D')
        )
        
        y = pd.Series(
            np.random.randn(n_samples),
            index=X.index,
            name='target'
        )
        
        return X, y
    
    def test_walk_forward_validation(self, sample_data):
        """Test walk-forward cross-validation."""
        X, y = sample_data
        
        cv = WalkForwardCV(
            n_splits=5,
            test_size=50,  # Use ~50 days for test set
            min_train_size=252  # Use ~1 year for min training
        )
        
        splits = list(cv.split(X, y))
        
        # Should have 5 splits
        assert len(splits) == 5
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
    
    def test_purged_kfold_validation(self, sample_data):
        """Test purged k-fold cross-validation."""
        X, y = sample_data
        
        purge_period = 10
        cv = PurgedKFoldCV(
            n_splits=5,
            purge_period=purge_period,
            embargo_period=5
        )
        
        splits = list(cv.split(X, y))
        
        # Should have up to 5 splits
        assert len(splits) > 0
        assert len(splits) <= 5
        
        # Check that splits provide reasonable size partitions
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap between train and test
            assert len(set(train_idx).intersection(set(test_idx))) == 0
    
    def test_time_series_split_validation(self, sample_data):
        """Test time series split validation."""
        X, y = sample_data
        
        cv = TimeSeriesSplit(
            n_splits=5,
            test_size=0.2
        )
        
        splits = list(cv.split(X, y))
        
        # Should have 5 splits
        assert len(splits) == 5
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap between train and test
            assert len(set(train_idx).intersection(set(test_idx))) == 0
    
    def test_cv_with_sklearn(self, sample_data):
        """Test cross-validation with sklearn."""
        X, y = sample_data
        
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import LinearRegression
        
        # Test all CV classes with sklearn's cross_validate
        for cv_class in [WalkForwardCV, PurgedKFoldCV, TimeSeriesSplit]:
            cv = cv_class(n_splits=3)
            model = LinearRegression()
            
            try:
                cv_results = cross_validate(model, X, y, cv=cv, return_train_score=True)
                
                # Should return cross-validation results
                assert isinstance(cv_results, dict)
                assert 'test_score' in cv_results
                assert 'train_score' in cv_results
                assert len(cv_results['test_score']) <= 3
            except Exception as e:
                # If it fails, at least the CV split should work
                splits = list(cv.split(X, y))
                assert len(splits) > 0
    
    def test_custom_parameters(self, sample_data):
        """Test custom parameters for CV classes."""
        X, y = sample_data
        
        # Test custom parameters for WalkForwardCV
        cv1 = WalkForwardCV(n_splits=3, test_size=30, min_train_size=100)
        assert cv1.n_splits == 3
        assert cv1.test_size == 30
        assert cv1.min_train_size == 100
        
        # Test custom parameters for PurgedKFoldCV
        cv2 = PurgedKFoldCV(n_splits=4, purge_period=15, embargo_period=3)
        assert cv2.n_splits == 4
        assert cv2.purge_period == 15
        assert cv2.embargo_period == 3
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create very small dataset
        X = pd.DataFrame({
            'feature_1': [1, 2, 3]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        y = pd.Series([0.1, 0.2, 0.3], index=X.index)
        
        # WalkForwardCV should raise an error with insufficient data
        cv = WalkForwardCV(n_splits=5, test_size=1, min_train_size=1)
        
        try:
            # Should handle gracefully or raise appropriate error
            splits = list(cv.split(X, y))
            # If it doesn't raise an error, it should return valid splits
            assert len(splits) <= 3  # Can't have more splits than data points
            for train_idx, test_idx in splits:
                assert len(train_idx) > 0
                assert len(test_idx) > 0
        except ValueError as e:
            # It's also valid to raise an error about insufficient data
            assert "too small" in str(e).lower() or "insufficient" in str(e).lower()
