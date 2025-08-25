"""
Data API for loading and managing financial data.

This module provides a unified interface for loading OHLCV data,
fundamentals, and other financial data sources with Parquet storage
and JSON metadata cataloging.
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pyarrow as pa
import pyarrow.parquet as pq


class DataCatalog:
    """Catalog for managing data metadata and file locations."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.catalog_file = self.data_dir / "catalog.json"
        self.catalog = self._load_catalog()
        
        # Ensure directories exist
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.curated_dir = self.data_dir / "curated"
        
        for dir_path in [self.raw_dir, self.interim_dir, self.curated_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_catalog(self) -> Dict[str, Any]:
        """Load existing catalog or create new one."""
        if self.catalog_file.exists():
            with open(self.catalog_file, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "datasets": {},
            "sources": {}
        }
    
    def _save_catalog(self):
        """Save catalog to disk."""
        self.catalog["updated_at"] = datetime.now().isoformat()
        with open(self.catalog_file, 'w') as f:
            json.dump(self.catalog, f, indent=2)
    
    def register_dataset(self, name: str, metadata: Dict[str, Any]):
        """Register a dataset in the catalog."""
        self.catalog["datasets"][name] = {
            **metadata,
            "registered_at": datetime.now().isoformat()
        }
        self._save_catalog()
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered dataset."""
        return self.catalog["datasets"].get(name)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self.catalog["datasets"].keys())


class DataAPI:
    """Main data API for loading and managing financial data."""
    
    def __init__(self, data_dir: str = "data", alpha_vantage_key: Optional[str] = None):
        self.catalog = DataCatalog(data_dir)
        self.alpha_vantage_key = alpha_vantage_key
        
        # Initialize data sources
        if alpha_vantage_key:
            self.alpha_vantage = TimeSeries(alpha_vantage_key)
        else:
            self.alpha_vantage = None
    
    def load_ohlcv(
        self, 
        tickers: Union[str, List[str]], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "yfinance",
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load OHLCV data for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('yfinance', 'alpha_vantage', 'csv')
            force_reload: Force reload from source instead of using cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Check cache first
        if not force_reload:
            cached_data = self._load_cached_ohlcv(tickers, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Load from source
        if source == "yfinance":
            data = self._load_from_yfinance(tickers, start_date, end_date)
        elif source == "alpha_vantage":
            data = self._load_from_alpha_vantage(tickers, start_date, end_date)
        elif source == "csv":
            data = self._load_from_csv(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Cache the data
        self._cache_ohlcv(data, tickers, start_date, end_date)
        
        # Register in catalog
        self._register_ohlcv_dataset(tickers, start_date, end_date, source)
        
        return data
    
    def _load_from_yfinance(
        self, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        data_list = []
        
        for ticker in tickers:
            try:
                # Download data
                ticker_data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if not ticker_data.empty:
                    # Add ticker column
                    ticker_data['ticker'] = ticker
                    data_list.append(ticker_data)
                
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
                continue
        
        if not data_list:
            raise ValueError("No data could be loaded for any ticker")
        
        # Combine all tickers
        combined_data = pd.concat(data_list, axis=0)
        combined_data = combined_data.sort_index()
        
        # Ensure proper column names
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'ticker']
        for col in expected_columns:
            if col not in combined_data.columns:
                if col == 'Adj Close':
                    combined_data[col] = combined_data['Close']
                elif col == 'Volume':
                    combined_data[col] = 0
        
        return combined_data
    
    def _load_from_alpha_vantage(
        self, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data from Alpha Vantage."""
        if not self.alpha_vantage:
            raise ValueError("Alpha Vantage API key not provided")
        
        data_list = []
        
        for ticker in tickers:
            try:
                # Get daily data
                data, meta = self.alpha_vantage.get_daily(symbol=ticker, outputsize='full')
                
                if data:
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Filter by date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    # Add ticker column
                    df['ticker'] = ticker
                    df['Adj Close'] = df['Close']  # Alpha Vantage doesn't provide adjusted close
                    
                    data_list.append(df)
                
            except Exception as e:
                print(f"Error loading {ticker} from Alpha Vantage: {e}")
                continue
        
        if not data_list:
            raise ValueError("No data could be loaded from Alpha Vantage")
        
        # Combine all tickers
        combined_data = pd.concat(data_list, axis=0)
        combined_data = combined_data.sort_index()
        
        return combined_data
    
    def _load_from_csv(
        self, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data from CSV files."""
        data_list = []
        
        for ticker in tickers:
            csv_path = self.catalog.raw_dir / f"{ticker}.csv"
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    
                    # Filter by date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    # Add ticker column
                    df['ticker'] = ticker
                    
                    data_list.append(df)
                    
                except Exception as e:
                    print(f"Error loading {ticker} from CSV: {e}")
                    continue
        
        if not data_list:
            raise ValueError("No CSV data found for any ticker")
        
        # Combine all tickers
        combined_data = pd.concat(data_list, axis=0)
        combined_data = combined_data.sort_index()
        
        return combined_data
    
    def _cache_ohlcv(
        self, 
        data: pd.DataFrame, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ):
        """Cache OHLCV data to Parquet file."""
        # Create filename
        ticker_str = "_".join(tickers)
        start_str = start_date.strftime("%Y%m%d") if start_date else "start"
        end_str = end_date.strftime("%Y%m%d") if end_date else "end"
        
        filename = f"ohlcv_{ticker_str}_{start_str}_{end_str}.parquet"
        filepath = self.catalog.interim_dir / filename
        
        # Save to Parquet
        data.to_parquet(filepath, engine='pyarrow')
    
    def _load_cached_ohlcv(
        self, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Optional[pd.DataFrame]:
        """Load cached OHLCV data if available."""
        # Create filename
        ticker_str = "_".join(tickers)
        start_str = start_date.strftime("%Y%m%d") if start_date else "start"
        end_str = end_date.strftime("%Y%m%d") if end_date else "end"
        
        filename = f"ohlcv_{ticker_str}_{start_str}_{end_str}.parquet"
        filepath = self.catalog.interim_dir / filename
        
        if filepath.exists():
            try:
                return pd.read_parquet(filepath)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return None
        
        return None
    
    def _register_ohlcv_dataset(
        self, 
        tickers: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        source: str
    ):
        """Register OHLCV dataset in catalog."""
        dataset_name = f"ohlcv_{'_'.join(tickers)}"
        
        metadata = {
            "type": "ohlcv",
            "tickers": tickers,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "source": source,
            "columns": ["Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"],
            "file_path": f"interim/ohlcv_{'_'.join(tickers)}_{start_date.strftime('%Y%m%d') if start_date else 'start'}_{end_date.strftime('%Y%m%d') if end_date else 'end'}.parquet"
        }
        
        self.catalog.register_dataset(dataset_name, metadata)
    
    def get_trading_calendar(
        self, 
        start_date: datetime, 
        end_date: datetime,
        market: str = "US"
    ) -> pd.DatetimeIndex:
        """
        Get trading calendar for specified period.
        
        Args:
            start_date: Start date
            end_date: End date
            market: Market identifier
            
        Returns:
            DatetimeIndex of trading days
        """
        # For US market, use business days
        if market == "US":
            calendar = pd.bdate_range(start=start_date, end=end_date, freq='B')
        else:
            # Default to business days
            calendar = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        return calendar
    
    def align_to_calendar(
        self, 
        data: pd.DataFrame, 
        calendar: pd.DatetimeIndex,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Align data to trading calendar.
        
        Args:
            data: Input data
            calendar: Trading calendar
            method: Forward fill method
            
        Returns:
            Aligned data
        """
        # Reindex to calendar
        aligned_data = data.reindex(calendar, method=method)
        
        # Forward fill missing values
        if method == "ffill":
            aligned_data = aligned_data.fillna(method='ffill')
        
        return aligned_data
    
    def calculate_returns(
        self, 
        data: pd.DataFrame, 
        price_col: str = "Adj Close",
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            data: Price data
            price_col: Column to use for price
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            DataFrame with returns
        """
        returns_data = data.copy()
        
        # Group by ticker and calculate returns
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker]
            ticker_prices = ticker_data[price_col]
            
            if method == "log":
                returns = np.log(ticker_prices / ticker_prices.shift(1))
            else:  # simple
                returns = ticker_prices.pct_change()
            
            returns_data.loc[ticker_data.index, 'returns'] = returns
        
        return returns_data
    
    def get_survivorship_bias_free_universe(
        self, 
        base_universe: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """
        Get survivorship-bias-free universe by checking data availability.
        
        Args:
            base_universe: Base list of tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            List of tickers with sufficient data
        """
        available_tickers = []
        
        for ticker in base_universe:
            try:
                # Try to load data
                data = self.load_ohlcv(ticker, start_date, end_date)
                
                # Check if we have sufficient data
                if len(data) > 252:  # At least 1 year
                    available_tickers.append(ticker)
                
            except Exception:
                continue
        
        return available_tickers
    
    def save_data(
        self, 
        data: pd.DataFrame, 
        name: str, 
        category: str = "interim",
        format: str = "parquet"
    ):
        """
        Save data to disk.
        
        Args:
            data: Data to save
            name: Dataset name
            category: Data category (raw, interim, curated)
            format: File format (parquet, csv)
        """
        if category == "raw":
            save_dir = self.catalog.raw_dir
        elif category == "interim":
            save_dir = self.catalog.interim_dir
        elif category == "curated":
            save_dir = self.catalog.curated_dir
        else:
            raise ValueError(f"Invalid category: {category}")
        
        filepath = save_dir / f"{name}.{format}"
        
        if format == "parquet":
            data.to_parquet(filepath, engine='pyarrow')
        elif format == "csv":
            data.to_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Register in catalog
        metadata = {
            "type": "custom",
            "name": name,
            "category": category,
            "format": format,
            "file_path": str(filepath),
            "shape": data.shape,
            "columns": list(data.columns)
        }
        
        self.catalog.register_dataset(name, metadata)
    
    def load_data(self, name: str, category: str = "interim") -> Optional[pd.DataFrame]:
        """
        Load data by name.
        
        Args:
            name: Dataset name
            category: Data category
            
        Returns:
            Loaded data or None if not found
        """
        dataset_info = self.catalog.get_dataset_info(name)
        
        if not dataset_info:
            return None
        
        file_path = dataset_info.get("file_path")
        if not file_path:
            return None
        
        # Determine full path
        if category == "raw":
            full_path = self.catalog.raw_dir / Path(file_path).name
        elif category == "interim":
            full_path = self.catalog.interim_dir / Path(file_path).name
        elif category == "curated":
            full_path = self.catalog.curated_dir / Path(file_path).name
        else:
            return None
        
        if not full_path.exists():
            return None
        
        try:
            if full_path.suffix == ".parquet":
                return pd.read_parquet(full_path)
            elif full_path.suffix == ".csv":
                return pd.read_csv(full_path, index_col=0, parse_dates=True)
            else:
                return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
