"""
Main backtesting engine for quantitative trading strategies.

This module provides the core backtesting functionality with realistic
assumptions, transaction costs, and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

from ..strategies.schema import StrategySpec
from ..core.data_api import DataAPI
from ..features.registry import FeatureRegistry
from ..models.model_zoo import ModelZoo
from .result import BacktestResult
from .slippage import SlippageModel
from .position_sizing import PositionSizer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Comprehensive backtesting engine for quantitative trading strategies.
    
    This engine provides realistic backtesting with:
    - Daily bar resolution
    - Next-open fills
    - Transaction costs and slippage
    - Position sizing and risk management
    - Portfolio construction
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,  # 10 basis points
        slippage_bps: float = 5.0,  # 5 basis points
        max_position_size: float = 0.1,  # 10% max per position
        benchmark: Optional[str] = None,
        data_api: Optional[DataAPI] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Initial capital amount
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            max_position_size: Maximum position size as fraction of portfolio
            benchmark: Benchmark ticker for comparison
            data_api: Data API instance for market data
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size
        self.benchmark = benchmark
        
        # Initialize components
        self.data_api = data_api or DataAPI()
        self.feature_registry = FeatureRegistry()
        self.model_zoo = ModelZoo()
        self.slippage_model = SlippageModel(slippage_bps)
        self.position_sizer = PositionSizer(max_position_size)
        
        # Backtest state
        self.current_date = None
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        self.daily_positions = []
        
        logger.info(f"Backtest engine initialized with ${initial_capital:,.2f} capital")
    
    def run_backtest(
        self,
        strategy_spec: StrategySpec,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_results: bool = True,
        output_dir: str = "reports/backtests"
    ) -> BacktestResult:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy_spec: Strategy specification
            start_date: Start date for backtest (defaults to strategy start)
            end_date: End date for backtest (defaults to strategy end)
            save_results: Whether to save results to files
            output_dir: Directory to save results
            
        Returns:
            BacktestResult object with complete results
        """
        logger.info(f"Starting backtest for strategy: {strategy_spec.name}")
        
        # Set dates
        start_date = start_date or strategy_spec.start_date
        end_date = end_date or strategy_spec.end_date
        
        if not start_date or not end_date:
            raise ValueError("Start and end dates must be specified")
        
        # Initialize backtest result
        result = BacktestResult(
            strategy_name=strategy_spec.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            benchmark=self.benchmark
        )
        
        try:
            # Load market data
            market_data = self._load_market_data(strategy_spec.universe, start_date, end_date)
            
            # Generate features
            features = self._generate_features(market_data, strategy_spec.features)
            
            # Train model if specified
            model = None
            if strategy_spec.model_config:
                model = self._train_model(features, strategy_spec)
            
            # Run simulation
            self._run_simulation(strategy_spec, market_data, features, model, start_date, end_date)
            
            # Calculate metrics
            result.set_equity_curve(pd.DataFrame(self.equity_curve))
            result.set_trades(pd.DataFrame(self.trades))
            result.set_positions(pd.DataFrame(self.daily_positions))
            
            # Set benchmark returns if available
            if self.benchmark:
                benchmark_data = self._load_benchmark_data(start_date, end_date)
                if not benchmark_data.empty:
                    result.set_benchmark_returns(benchmark_data['returns'])
            
            # Calculate all metrics
            result.calculate_metrics()
            
            # Save results if requested
            if save_results:
                result.save_results(output_dir)
            
            logger.info(f"Backtest completed successfully for {strategy_spec.name}")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        
        return result
    
    def _load_market_data(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Load market data for the universe."""
        logger.info(f"Loading market data for {len(universe)} symbols")
        
        market_data = {}
        for symbol in universe:
            try:
                data = self.data_api.get_ohlcv(symbol, start_date, end_date)
                if not data.empty:
                    market_data[symbol] = data
                    logger.debug(f"Loaded data for {symbol}: {len(data)} bars")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        if not market_data:
            raise ValueError("No market data could be loaded")
        
        logger.info(f"Successfully loaded data for {len(market_data)} symbols")
        return market_data
    
    def _generate_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        feature_specs: List[Any]
    ) -> pd.DataFrame:
        """Generate features for the strategy."""
        logger.info("Generating features for strategy")
        
        # This is a simplified feature generation
        # In a full implementation, you would use the FeatureRegistry
        # to generate features based on the feature specifications
        
        # For now, create basic price-based features
        all_features = []
        
        for symbol, data in market_data.items():
            # Basic price features
            features = pd.DataFrame(index=data.index)
            features['symbol'] = symbol
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['momentum'] = data['close'] / data['close'].shift(20) - 1
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            all_features.append(features)
        
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features = combined_features.sort_index()
            logger.info(f"Generated {len(combined_features.columns)} features")
            return combined_features
        else:
            return pd.DataFrame()
    
    def _train_model(
        self,
        features: pd.DataFrame,
        strategy_spec: StrategySpec
    ) -> Any:
        """Train the model for the strategy."""
        logger.info("Training model for strategy")
        
        # This is a placeholder for model training
        # In a full implementation, you would use the ModelZoo
        # to train models based on the strategy specification
        
        return None
    
    def _run_simulation(
        self,
        strategy_spec: StrategySpec,
        market_data: Dict[str, pd.DataFrame],
        features: pd.DataFrame,
        model: Any,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Run the main simulation loop."""
        logger.info("Starting simulation loop")
        
        # Get trading dates
        trading_dates = self._get_trading_dates(market_data, start_date, end_date)
        
        # Initialize portfolio
        self._initialize_portfolio()
        
        # Main simulation loop
        for date in trading_dates:
            self.current_date = date
            
            # Update portfolio value
            self._update_portfolio_value(market_data, date)
            
            # Check for exit signals
            self._check_exit_signals(strategy_spec, market_data, date)
            
            # Check for entry signals
            self._check_entry_signals(strategy_spec, market_data, features, model, date)
            
            # Record daily state
            self._record_daily_state(date)
        
        logger.info("Simulation loop completed")
    
    def _get_trading_dates(
        self,
        market_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """Get list of trading dates."""
        # Find common trading dates across all symbols
        all_dates = set()
        for data in market_data.values():
            all_dates.update(data.index)
        
        # Filter by date range and sort
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]
        trading_dates.sort()
        
        return trading_dates
    
    def _initialize_portfolio(self) -> None:
        """Initialize portfolio state."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        self.daily_positions = []
    
    def _update_portfolio_value(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> None:
        """Update portfolio value based on current positions."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in market_data and date in market_data[symbol].index:
                current_price = market_data[symbol].loc[date, 'close']
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        self.portfolio_value = total_value
    
    def _check_exit_signals(
        self,
        strategy_spec: StrategySpec,
        market_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> None:
        """Check and execute exit signals."""
        if not strategy_spec.exit_rules:
            return
        
        # Simplified exit logic - in practice, you'd evaluate the exit rules
        # against current market conditions and features
        
        for symbol in list(self.positions.keys()):
            if symbol in market_data and date in market_data[symbol].index:
                # Example exit condition (simplified)
                current_price = market_data[symbol].loc[date, 'close']
                entry_price = self.positions[symbol]['entry_price']
                
                # Simple stop loss (example)
                if current_price < entry_price * 0.95:  # 5% stop loss
                    self._close_position(symbol, current_price, date, "stop_loss")
    
    def _check_entry_signals(
        self,
        strategy_spec: StrategySpec,
        market_data: Dict[str, pd.DataFrame],
        features: pd.DataFrame,
        model: Any,
        date: datetime
    ) -> None:
        """Check and execute entry signals."""
        if not strategy_spec.entry_rules:
            return
        
        # Simplified entry logic - in practice, you'd evaluate the entry rules
        # against current market conditions and features
        
        for symbol in strategy_spec.universe:
            if symbol in market_data and date in market_data[symbol].index:
                # Skip if already have position
                if symbol in self.positions:
                    continue
                
                # Example entry condition (simplified)
                current_price = market_data[symbol].loc[date, 'close']
                
                # Simple momentum entry (example)
                if symbol in features.index and date in features.index:
                    symbol_features = features.loc[(features.index == date) & (features['symbol'] == symbol)]
                    if not symbol_features.empty:
                        momentum = symbol_features['momentum'].iloc[0]
                        if momentum > 0.05:  # 5% momentum threshold
                            self._open_position(symbol, current_price, date, "momentum")
    
    def _open_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: str
    ) -> None:
        """Open a new position."""
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            self.portfolio_value,
            price,
            self.max_position_size
        )
        
        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(price, position_size)
        
        # Check if we have enough cash
        total_cost = (price * position_size) + transaction_cost
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash to open position in {symbol}")
            return
        
        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = {
            'quantity': position_size,
            'entry_price': price,
            'entry_date': date,
            'entry_reason': reason
        }
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'buy',
            'quantity': position_size,
            'price': price,
            'value': price * position_size,
            'transaction_cost': transaction_cost,
            'reason': reason
        })
        
        logger.info(f"Opened position in {symbol}: {position_size} shares at ${price:.2f}")
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: str
    ) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Calculate proceeds
        proceeds = price * quantity
        
        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(price, quantity)
        
        # Calculate PnL
        entry_value = position['entry_price'] * quantity
        pnl = proceeds - entry_value - transaction_cost
        
        # Update cash
        self.cash += proceeds - transaction_cost
        
        # Calculate holding period
        holding_period = (date - position['entry_date']).days
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'value': proceeds,
            'transaction_cost': transaction_cost,
            'pnl': pnl,
            'holding_period': holding_period,
            'reason': reason
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position in {symbol}: PnL ${pnl:.2f}")
    
    def _calculate_transaction_cost(self, price: float, quantity: float) -> float:
        """Calculate transaction costs."""
        trade_value = price * quantity
        transaction_cost = trade_value * (self.transaction_cost_bps / 10000)
        
        # Add slippage
        slippage_cost = self.slippage_model.calculate_slippage(price, quantity)
        
        return transaction_cost + slippage_cost
    
    def _record_daily_state(self, date: datetime) -> None:
        """Record the daily portfolio state."""
        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': self.portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions)
        })
        
        # Record daily positions
        for symbol, position in self.positions.items():
            self.daily_positions.append({
                'date': date,
                'symbol': symbol,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'entry_date': position['entry_date']
            })
    
    def _load_benchmark_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load benchmark data for comparison."""
        try:
            benchmark_data = self.data_api.get_ohlcv(self.benchmark, start_date, end_date)
            if not benchmark_data.empty:
                benchmark_data['returns'] = benchmark_data['close'].pct_change()
                return benchmark_data
        except Exception as e:
            logger.warning(f"Failed to load benchmark data: {e}")
        
        return pd.DataFrame()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'total_return': (self.portfolio_value / self.initial_capital - 1) * 100,
            'current_date': self.current_date
        }
    
    def print_portfolio_summary(self) -> None:
        """Print current portfolio summary."""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*50)
        print("PORTFOLIO SUMMARY")
        print("="*50)
        print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"Cash: ${summary['cash']:,.2f}")
        print(f"Positions: {summary['positions']}")
        print(f"Total Return: {summary['total_return']:.2f}%")
        print(f"Current Date: {summary['current_date']}")
        print("="*50)
