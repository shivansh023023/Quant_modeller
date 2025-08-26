#!/usr/bin/env python3
"""
Simple Custom Momentum Algorithm Demo
=====================================

This demonstrates how to create your own quantitative trading algorithm
with the platform. This version focuses on the core concepts without
complex technical indicator calculations.

Strategy: Simple Momentum + RSI
- Buy when 20-day returns > 5% and RSI < 70
- Sell when 20-day returns < 2% or RSI > 75
- Equal weight position sizing with basic risk management
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List


class SimpleMomentumStrategy:
    """
    Simple momentum trading strategy for demonstration
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        
        # Strategy parameters
        self.momentum_threshold = 0.05  # 5% momentum threshold
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = data.copy()
        
        # Flatten column names if needed (yfinance issue)
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        
        # Calculate indicators
        df['returns_20d'] = df['Close'].pct_change(20)
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['sma_20'] = df['Close'].rolling(20).mean()
        
        # Initialize signals
        df['signal'] = 0
        
        # Buy conditions: Strong momentum + not overbought + above SMA
        buy_mask = (
            (df['returns_20d'] > self.momentum_threshold) &
            (df['rsi'] < self.rsi_overbought) &
            (df['Close'] > df['sma_20'])
        )
        df.loc[buy_mask, 'signal'] = 1
        
        # Sell conditions: Weak momentum or overbought
        sell_mask = (
            (df['returns_20d'] < 0.02) |
            (df['rsi'] > 75)
        )
        df.loc[sell_mask, 'signal'] = -1
        
        return df
    
    def backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run backtest"""
        print("🚀 Running Simple Momentum Algorithm Backtest")
        print("="*60)
        
        # Download data
        all_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 100:
                    all_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} trading days")
            except Exception as e:
                print(f"❌ {symbol}: {e}")
        
        print(f"\n📊 Processing {len(all_data)} symbols...")
        
        # Generate signals
        signals_data = {}
        for symbol, data in all_data.items():
            signals = self.generate_signals(data)
            signals_data[symbol] = signals
            
            buy_count = (signals['signal'] == 1).sum()
            sell_count = (signals['signal'] == -1).sum()
            print(f"📈 {symbol}: {buy_count} buy, {sell_count} sell signals")
        
        # Simulate trading
        print("\n💼 Simulating portfolio...")
        results = self.simulate_trades(signals_data)
        
        return results
    
    def simulate_trades(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Simulate trading with the signals"""
        # Get all dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        trading_dates = sorted(all_dates)
        
        portfolio_values = []
        returns_list = []
        
        for date in trading_dates:
            # Calculate current portfolio value
            portfolio_value = self.cash
            for symbol, shares in self.positions.items():
                if symbol in data and date in data[symbol].index:
                    price = data[symbol].loc[date, 'Close']
                    portfolio_value += shares * price
            
            portfolio_values.append(portfolio_value)
            
            # Calculate return
            if len(portfolio_values) > 1:
                daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                returns_list.append(daily_return)
            
            # Process signals
            for symbol, df in data.items():
                if date not in df.index:
                    continue
                
                signal = df.loc[date, 'signal']
                price = df.loc[date, 'Close']
                
                if signal == 1 and symbol not in self.positions:
                    # Buy signal
                    position_size = self.cash / 8  # Max 8 positions
                    shares = int(position_size / price)
                    if shares > 0:
                        cost = shares * price
                        if cost <= self.cash:
                            self.cash -= cost
                            self.positions[symbol] = shares
                            self.trades.append({
                                'date': date, 'symbol': symbol, 'action': 'BUY',
                                'shares': shares, 'price': price
                            })
                
                elif signal == -1 and symbol in self.positions:
                    # Sell signal
                    shares = self.positions[symbol]
                    proceeds = shares * price
                    self.cash += proceeds
                    del self.positions[symbol]
                    self.trades.append({
                        'date': date, 'symbol': symbol, 'action': 'SELL',
                        'shares': shares, 'price': price
                    })
        
        # Calculate metrics
        portfolio_series = pd.Series(portfolio_values, index=trading_dates)
        returns_series = pd.Series(returns_list, index=trading_dates[1:])
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        if len(returns_series) > 0:
            annual_return = returns_series.mean() * 252
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
            max_dd = self.calculate_max_drawdown(portfolio_series)
            win_rate = (returns_series > 0).mean()
        else:
            annual_return = sharpe = max_dd = win_rate = 0
        
        return {
            'portfolio_values': portfolio_series,
            'returns': returns_series,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(self.trades)
        }
    
    def calculate_max_drawdown(self, portfolio: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak
        return abs(drawdown.min())
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Simple Momentum Algorithm - Backtest Results', fontsize=16, fontweight='bold')
        
        # Equity curve
        ax1.plot(results['portfolio_values'], color='blue', linewidth=2)
        ax1.axhline(self.initial_capital, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Portfolio Value')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        portfolio = results['portfolio_values']
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        if len(results['returns']) > 0:
            ax3.hist(results['returns'] * 100, bins=50, alpha=0.7, edgecolor='black')
            ax3.axvline(results['returns'].mean() * 100, color='red', linestyle='--')
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        if len(results['returns']) > 60:
            rolling_sharpe = (results['returns'].rolling(60).mean() / 
                            results['returns'].rolling(60).std() * np.sqrt(252))
            ax4.plot(rolling_sharpe, color='purple', linewidth=2)
            ax4.axhline(1, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Rolling Sharpe Ratio (60-day)')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig('simple_momentum_results.png', dpi=300, bbox_inches='tight')
        print("📊 Chart saved as 'simple_momentum_results.png'")


def main():
    """Main execution function"""
    print("🎯 SIMPLE MOMENTUM ALGORITHM DEMONSTRATION")
    print("="*60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')  # 3 years
    
    print(f"📊 Symbols: {symbols}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Capital: $100,000")
    
    # Run backtest
    strategy = SimpleMomentumStrategy(initial_capital=100000)
    results = strategy.backtest(symbols, start_date, end_date)
    
    # Display results
    if results and len(results['portfolio_values']) > 0:
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        print(f"💰 Final Value: ${results['final_value']:,.2f}")
        print(f"📈 Total Return: {results['total_return']:.2%}")
        print(f"📅 Annual Return: {results['annual_return']:.2%}")
        print(f"⚡ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {results['win_rate']:.2%}")
        print(f"🔄 Total Trades: {results['total_trades']}")
        
        profit = results['final_value'] - strategy.initial_capital
        print(f"💵 Profit/Loss: ${profit:,.2f}")
        
        # Performance assessment
        if results['sharpe_ratio'] > 1.0:
            print("\n✅ EXCELLENT: Sharpe > 1.0 indicates strong performance!")
        elif results['sharpe_ratio'] > 0.5:
            print("\n✅ GOOD: Positive risk-adjusted returns")
        else:
            print("\n⚠️ NEEDS IMPROVEMENT: Consider parameter optimization")
        
        # Create visualization
        print("\n📊 Creating performance charts...")
        strategy.plot_results(results)
        
        print("\n🎉 ALGORITHM DEMONSTRATION COMPLETE!")
        print("="*60)
        print("📋 STRATEGY OVERVIEW:")
        print("• Simple momentum + RSI filtering")
        print("• Equal weight position sizing")
        print("• Basic risk management rules")
        print("• Realistic performance simulation")
        
        print("\n💡 CUSTOMIZATION IDEAS:")
        print("1. Add more technical indicators")
        print("2. Implement dynamic position sizing")
        print("3. Add stop-loss and take-profit rules")
        print("4. Test different momentum thresholds")
        print("5. Experiment with different stock universes")
        
        return results
    else:
        print("❌ Backtest failed - no results generated")
        return None


if __name__ == "__main__":
    results = main()
