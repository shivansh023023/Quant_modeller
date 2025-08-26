#!/usr/bin/env python3
"""
Custom Algorithm Creation: Smart Momentum Strategy
==================================================

This script demonstrates how to create your own quantitative trading algorithm
using the AI-Powered Quantitative Trading Platform.

Strategy Idea: "Buy stocks showing strong momentum but avoid overly volatile periods"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data_api import DataAPI
from src.features.generators import FeatureGenerator
from src.backtest.engine import BacktestEngine
from src.viz.dashboard import Dashboard
from src.tracking.experiment_manager import ExperimentManager
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_smart_momentum_algorithm():
    """
    Create a custom Smart Momentum trading algorithm
    """
    print("🚀 Creating Custom Smart Momentum Algorithm...")
    print("="*60)
    
    # Step 1: Define our trading idea in natural language
    trading_idea = """
    I want to create a momentum strategy that:
    1. Buys stocks when they show strong price momentum (20-day returns > 5%)
    2. Uses RSI to avoid overbought conditions (RSI < 70)
    3. Filters out highly volatile stocks (volatility < 30%)
    4. Uses volume confirmation (volume above 20-day average)
    5. Exits when momentum weakens or RSI becomes overbought
    """
    
    print("📝 Trading Idea:")
    print(trading_idea)
    print("\n" + "="*60)
    
    # Step 2: Define Strategy Configuration
    strategy_config = {
        "name": "Smart Momentum Strategy",
        "description": "Multi-factor momentum strategy with volatility filtering",
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
        "start_date": "2020-01-01",
        "end_date": "2024-08-25",
        "initial_capital": 100000,
        "position_sizing": "equal_weight",
        "commission": 0.001,
        "slippage": 0.0005,
        "stop_loss": 0.05,
        "take_profit": 0.15
    }
    
    print("⚙️ Strategy Configuration Created!")
    print(f"📊 Testing on {len(strategy_config['symbols'])} stocks")
    print(f"📅 Period: {strategy_config['start_date']} to {strategy_config['end_date']}")
    print(f"💰 Initial Capital: ${strategy_config['initial_capital']:,}")
    
    # Step 3: Set up experiment tracking
    print("\n📊 Setting up experiment tracking...")
    try:
        experiment_manager = ExperimentManager(base_dir="my_custom_algorithms")
        
        # Create experiment
        experiment_id = experiment_manager.create_experiment(
            strategy_name=strategy_config["name"],
            description="Custom Smart Momentum Algorithm",
            strategy_config=strategy_config
        )
        
        print(f"✅ Experiment created with ID: {experiment_id}")
        print(f"📁 Results will be saved to: my_custom_algorithms/strategies/")
        
    except Exception as e:
        print(f"⚠️ Experiment tracking setup failed: {e}")
        experiment_id = f"custom_momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"💡 Using fallback experiment ID: {experiment_id}")
    
    # Step 4: Display algorithm logic
    print("\n📋 YOUR CUSTOM ALGORITHM LOGIC:")
    print("="*40)
    print("BUY SIGNALS:")
    print("- 20-day momentum > 5%")
    print("- RSI < 70 (not overbought)")
    print("- 20-day volatility < 30%") 
    print("- Volume > 20-day average")
    print()
    print("SELL SIGNALS:")
    print("- 20-day momentum < 2%")
    print("- RSI > 75 (overbought)")
    print("- Stop loss: -5%")
    print("- Take profit: +15%")
    print()
    print("POSITION SIZING:")
    print("- Equal weight across positions")
    print("- Max positions based on available capital")
    print("- Risk management with stop losses")
    
    # Step 5: Display next steps
    print("\n" + "="*60)
    print("🎯 NEXT STEPS TO COMPLETE YOUR ALGORITHM:")
    print("="*60)
    print("1. Run the full backtesting pipeline:")
    print("   python run_custom_backtest.py")
    print()
    print("2. The backtest will automatically:")
    print("   - Collect market data")
    print("   - Generate 100+ technical features")
    print("   - Create trading signals")
    print("   - Simulate realistic trading")
    print("   - Calculate performance metrics")
    print("   - Generate visualizations")
    
    return strategy_config, experiment_id

def create_backtest_runner():
    """Create a script to run the complete backtest"""
    
    backtest_code = '''#!/usr/bin/env python3
"""
Run Custom Smart Momentum Algorithm Backtest
===========================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data_api import DataAPI
from src.features.generators import FeatureGenerator
from src.models.model_zoo import ModelZoo
from src.backtest.engine import BacktestEngine
from src.viz.dashboard import Dashboard
from src.tracking.experiment_manager import ExperimentManager
import pandas as pd

def run_complete_backtest():
    """Run the complete algorithm backtest pipeline"""
    
    print("🚀 Running Smart Momentum Algorithm Backtest...")
    
    # 1. Data Collection
    print("📊 Collecting market data...")
    data_api = DataAPI()
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    
    market_data = {}
    for symbol in symbols:
        try:
            market_data[symbol] = data_api.get_stock_data(
                symbol=symbol,
                start_date="2020-01-01",
                end_date="2024-08-25"
            )
            print(f"✅ {symbol}: {len(market_data[symbol])} days of data")
        except Exception as e:
            print(f"❌ {symbol}: Failed to fetch data - {e}")
    
    # 2. Feature Engineering
    print("\\n🔧 Generating features...")
    feature_gen = FeatureGenerator()
    
    all_features = {}
    for symbol, data in market_data.items():
        if data is not None and len(data) > 100:
            features = feature_gen.generate_all_features(data)
            all_features[symbol] = features
            print(f"✅ {symbol}: Generated {features.shape[1]} features")
    
    # 3. Create Trading Signals
    print("\\n🎯 Generating trading signals...")
    signals = {}
    
    for symbol, features in all_features.items():
        # Smart Momentum Logic
        momentum_20d = features['returns_20d'] * 100  # Convert to percentage
        rsi_14 = features.get('rsi_14', 50)
        volatility_20d = features['volatility_20d'] * 100
        volume_ratio = features.get('volume_ratio_20d', 1)
        
        # Buy conditions
        buy_signal = (
            (momentum_20d > 5) &           # Strong momentum
            (rsi_14 < 70) &                # Not overbought
            (volatility_20d < 30) &        # Not too volatile
            (volume_ratio > 1.1)           # Above average volume
        )
        
        # Sell conditions  
        sell_signal = (
            (momentum_20d < 2) |           # Weakening momentum
            (rsi_14 > 75)                  # Overbought
        )
        
        # Create signal series (-1: sell, 0: hold, 1: buy)
        signal_series = pd.Series(0, index=features.index)
        signal_series[buy_signal] = 1
        signal_series[sell_signal] = -1
        
        signals[symbol] = signal_series
        
        buy_days = (signal_series == 1).sum()
        sell_days = (signal_series == -1).sum()
        print(f"✅ {symbol}: {buy_days} buy signals, {sell_days} sell signals")
    
    # 4. Run Backtest
    print("\\n📈 Running backtest simulation...")
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    results = backtest_engine.run_backtest(
        market_data=market_data,
        signals=signals,
        position_sizing="equal_weight"
    )
    
    # 5. Display Results
    print("\\n📊 BACKTEST RESULTS:")
    print("="*50)
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}") 
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Total Trades: {results.total_trades}")
    
    # 6. Create Visualizations
    print("\\n📊 Creating visualizations...")
    dashboard = Dashboard()
    
    # Generate charts
    dashboard.create_equity_curve(results.equity_curve)
    dashboard.create_drawdown_chart(results.drawdown_series)
    dashboard.create_monthly_returns_heatmap(results.returns_series)
    
    print("✅ Backtest completed successfully!")
    print("📁 Check the 'my_custom_algorithms' folder for detailed results")

if __name__ == "__main__":
    run_complete_backtest()
'''
    
    with open("run_custom_backtest.py", "w") as f:
        f.write(backtest_code)
    
    print("📝 Created run_custom_backtest.py")

if __name__ == "__main__":
    # Create the custom algorithm
    strategy, exp_id = create_smart_momentum_algorithm()
    
    if strategy:
        print("\n🎉 SUCCESS! Your custom Smart Momentum algorithm has been created!")
        
        # Create additional helper scripts
        create_backtest_runner()
        
        print("\n💡 To test your algorithm:")
        print("   python run_custom_backtest.py")
    else:
        print("\n❌ Algorithm creation failed. Check the error messages above.")
