"""
Backtest result storage and analysis.

This module provides comprehensive storage and analysis of backtest results,
including performance metrics, trade logs, and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import logging

from ..core.metrics import PerformanceMetrics, RiskMetrics
from ..ai.explainability import StrategyExplainer

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Comprehensive backtest result storage and analysis.
    
    This class stores all backtest results and provides methods for
    analysis, visualization, and research note generation.
    """
    
    def __init__(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        benchmark: Optional[str] = None
    ):
        """
        Initialize backtest result.
        
        Args:
            strategy_name: Name of the strategy
            start_date: Start date of backtest
            end_date: End date of backtest
            initial_capital: Initial capital amount
            benchmark: Benchmark ticker for comparison
        """
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        
        # Core data storage
        self.equity_curve = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.daily_returns = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        
        # Performance metrics
        self.performance_metrics = {}
        self.risk_metrics = {}
        self.trade_metrics = {}
        
        # Metadata
        self.metadata = {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'benchmark': benchmark,
            'created_at': datetime.now(),
            'version': '1.0.0'
        }
        
        # Initialize metrics calculators
        self.performance_calculator = PerformanceMetrics()
        self.risk_calculator = RiskMetrics()
        
        # Initialize explainability
        self.explainer = StrategyExplainer()
    
    def set_equity_curve(self, equity_curve: pd.DataFrame) -> None:
        """
        Set the equity curve data.
        
        Args:
            equity_curve: DataFrame with columns ['date', 'equity', 'cash', 'positions']
        """
        self.equity_curve = equity_curve.copy()
        self.equity_curve['date'] = pd.to_datetime(self.equity_curve['date'])
        self.equity_curve.set_index('date', inplace=True)
        
        # Calculate daily returns
        self.daily_returns = self.equity_curve['equity'].pct_change().dropna()
    
    def set_trades(self, trades: pd.DataFrame) -> None:
        """
        Set the trades data.
        
        Args:
            trades: DataFrame with trade information
        """
        self.trades = trades.copy()
        if not self.trades.empty:
            self.trades['date'] = pd.to_datetime(self.trades['date'])
            self.trades.set_index('date', inplace=True)
    
    def set_positions(self, positions: pd.DataFrame) -> None:
        """
        Set the positions data.
        
        Args:
            positions: DataFrame with position information
        """
        self.positions = positions.copy()
        if not self.positions.empty:
            self.positions['date'] = pd.to_datetime(self.positions['date'])
            self.positions.set_index('date', inplace=True)
    
    def set_benchmark_returns(self, benchmark_returns: pd.Series) -> None:
        """
        Set benchmark returns for comparison.
        
        Args:
            benchmark_returns: Series of benchmark returns
        """
        self.benchmark_returns = benchmark_returns.copy()
        if not self.benchmark_returns.empty:
            self.benchmark_returns.index = pd.to_datetime(self.benchmark_returns.index)
    
    def calculate_metrics(self) -> None:
        """Calculate comprehensive performance and risk metrics."""
        if self.daily_returns.empty:
            logger.warning("No daily returns data available for metrics calculation")
            return
        
        # Performance metrics
        self.performance_metrics = self.performance_calculator.calculate_all_metrics(
            self.daily_returns,
            self.initial_capital
        )
        
        # Risk metrics
        self.risk_metrics = self.risk_calculator.calculate_all_metrics(
            self.daily_returns,
            self.benchmark_returns if not self.benchmark_returns.empty else None
        )
        
        # Trade metrics
        self.trade_metrics = self._calculate_trade_metrics()
        
        logger.info("All metrics calculated successfully")
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        if self.trades.empty:
            return {}
        
        metrics = {}
        
        # Basic trade counts
        metrics['total_trades'] = len(self.trades)
        metrics['winning_trades'] = len(self.trades[self.trades['pnl'] > 0])
        metrics['losing_trades'] = len(self.trades[self.trades['pnl'] < 0])
        
        # Win rate
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        # PnL metrics
        if 'pnl' in self.trades.columns:
            metrics['total_pnl'] = self.trades['pnl'].sum()
            metrics['avg_win'] = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_loss'] = self.trades[self.trades['pnl'] < 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0
            metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
        
        # Holding period metrics
        if 'holding_period' in self.trades.columns:
            metrics['avg_holding_period'] = self.trades['holding_period'].mean()
            metrics['max_holding_period'] = self.trades['holding_period'].max()
            metrics['min_holding_period'] = self.trades['holding_period'].min()
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the backtest results.
        
        Returns:
            Dictionary containing all key metrics and information
        """
        summary = {
            'strategy_info': {
                'name': self.strategy_name,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'final_capital': self.equity_curve['equity'].iloc[-1] if not self.equity_curve.empty else self.initial_capital,
                'benchmark': self.benchmark
            },
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_metrics,
            'trade_metrics': self.trade_metrics,
            'metadata': self.metadata
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the backtest results."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("="*60)
        
        # Strategy info
        print(f"\n📊 STRATEGY INFO:")
        print(f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${summary['strategy_info']['final_capital']:,.2f}")
        
        # Key performance metrics
        if self.performance_metrics:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"  Total Return: {self.performance_metrics.get('total_return', 0):.2%}")
            print(f"  Annualized Return: {self.performance_metrics.get('annualized_return', 0):.2%}")
            print(f"  Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {self.performance_metrics.get('max_drawdown', 0):.2%}")
        
        # Risk metrics
        if self.risk_metrics:
            print(f"\n⚠️  RISK METRICS:")
            print(f"  Volatility: {self.risk_metrics.get('volatility', 0):.2%}")
            print(f"  VaR (95%): {self.risk_metrics.get('var_95', 0):.2%}")
            print(f"  CVaR (95%): {self.risk_metrics.get('cvar_95', 0):.2%}")
        
        # Trade metrics
        if self.trade_metrics:
            print(f"\n🔄 TRADE METRICS:")
            print(f"  Total Trades: {self.trade_metrics.get('total_trades', 0)}")
            print(f"  Win Rate: {self.trade_metrics.get('win_rate', 0):.2%}")
            print(f"  Profit Factor: {self.trade_metrics.get('profit_factor', 0):.2f}")
        
        print("\n" + "="*60)
    
    def save_results(self, output_dir: str) -> None:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        if not self.equity_curve.empty:
            equity_file = output_path / f"{self.strategy_name}_equity_curve.csv"
            self.equity_curve.to_csv(equity_file)
            logger.info(f"Saved equity curve to: {equity_file}")
        
        # Save trades
        if not self.trades.empty:
            trades_file = output_path / f"{self.strategy_name}_trades.csv"
            self.trades.to_csv(trades_file)
            logger.info(f"Saved trades to: {trades_file}")
        
        # Save summary
        summary_file = output_path / f"{self.strategy_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        logger.info(f"Saved summary to: {summary_file}")
        
        # Save metadata
        metadata_file = output_path / f"{self.strategy_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to: {metadata_file}")
    
    def generate_research_notes(self, strategy_spec: Any = None) -> str:
        """
        Generate comprehensive research notes for the strategy.
        
        Args:
            strategy_spec: Strategy specification object
            
        Returns:
            Generated research notes as markdown
        """
        if strategy_spec:
            return self.explainer.generate_research_note(
                strategy_spec,
                self.get_summary(),
                self.performance_metrics
            )
        else:
            # Generate basic notes without strategy spec
            return self._generate_basic_notes()
    
    def _generate_basic_notes(self) -> str:
        """Generate basic research notes without strategy specification."""
        notes = f"""
# Backtest Research Notes: {self.strategy_name}

## Executive Summary
- **Strategy**: {self.strategy_name}
- **Period**: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
- **Initial Capital**: ${self.initial_capital:,.2f}
- **Final Capital**: ${self.get_summary()['strategy_info']['final_capital']:,.2f}

## Performance Analysis
- **Total Return**: {self.performance_metrics.get('total_return', 0):.2%}
- **Annualized Return**: {self.performance_metrics.get('annualized_return', 0):.2%}
- **Sharpe Ratio**: {self.performance_metrics.get('sharpe_ratio', 0):.2f}
- **Max Drawdown**: {self.performance_metrics.get('max_drawdown', 0):.2%}

## Risk Assessment
- **Volatility**: {self.risk_metrics.get('volatility', 0):.2%}
- **VaR (95%)**: {self.risk_metrics.get('var_95', 0):.2%}
- **CVaR (95%)**: {self.risk_metrics.get('cvar_95', 0):.2%}

## Trading Activity
- **Total Trades**: {self.trade_metrics.get('total_trades', 0)}
- **Win Rate**: {self.trade_metrics.get('win_rate', 0):.2%}
- **Profit Factor**: {self.trade_metrics.get('profit_factor', 0):.2f}

## Conclusions
This backtest demonstrates the strategy's performance over the specified period.
Further analysis and optimization may be warranted based on these results.
        """
        
        return notes.strip()
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot the equity curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.equity_curve.empty:
                logger.warning("No equity curve data available for plotting")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.plot(self.equity_curve.index, self.equity_curve['equity'], 
                    label='Strategy Equity', linewidth=2)
            
            # Plot benchmark if available
            if not self.benchmark_returns.empty:
                benchmark_equity = (1 + self.benchmark_returns).cumprod() * self.initial_capital
                plt.plot(benchmark_equity.index, benchmark_equity, 
                        label=f'Benchmark ({self.benchmark})', linewidth=2, alpha=0.7)
            
            plt.title(f'Equity Curve: {self.strategy_name}')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved equity curve plot to: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """
        Get detailed trade analysis.
        
        Returns:
            Dictionary containing trade analysis
        """
        if self.trades.empty:
            return {}
        
        analysis = {
            'trade_distribution': {
                'pnl_distribution': self.trades['pnl'].describe() if 'pnl' in self.trades.columns else {},
                'holding_period_distribution': self.trades['holding_period'].describe() if 'holding_period' in self.trades.columns else {},
            },
            'monthly_performance': self._calculate_monthly_performance(),
            'drawdown_analysis': self._calculate_drawdown_analysis(),
        }
        
        return analysis
    
    def _calculate_monthly_performance(self) -> pd.DataFrame:
        """Calculate monthly performance metrics."""
        if self.daily_returns.empty:
            return pd.DataFrame()
        
        monthly_returns = self.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_metrics = pd.DataFrame({
            'return': monthly_returns,
            'cumulative_return': (1 + monthly_returns).cumprod() - 1
        })
        
        return monthly_metrics
    
    def _calculate_drawdown_analysis(self) -> Dict[str, Any]:
        """Calculate detailed drawdown analysis."""
        if self.equity_curve.empty:
            return {}
        
        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        analysis = {
            'max_drawdown': drawdown.min(),
            'max_drawdown_date': drawdown.idxmin(),
            'drawdown_duration': self._calculate_drawdown_duration(drawdown),
            'drawdown_periods': len(drawdown[drawdown < 0])
        }
        
        return analysis
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate the duration of the maximum drawdown."""
        if drawdown.empty:
            return 0
        
        max_dd_date = drawdown.idxmin()
        recovery_date = drawdown[drawdown.index > max_dd_date]
        recovery_date = recovery_date[recovery_date >= 0]
        
        if recovery_date.empty:
            return len(drawdown) - drawdown.index.get_loc(max_dd_date)
        
        recovery_date = recovery_date.index[0]
        return (recovery_date - max_dd_date).days
