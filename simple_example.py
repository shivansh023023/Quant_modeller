#!/usr/bin/env python3
"""
Simple Example: How to Use Quant Lab

This script shows the exact workflow a user should follow:
1. Create a strategy
2. Test it with sample data
3. Analyze results
4. Generate visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main workflow example."""
    print("🚀 QUANT LAB - SIMPLE USER WORKFLOW")
    print("=" * 50)
    
    try:
        # STEP 1: Import what you need
        print("📦 Step 1: Importing modules...")
        from src import StrategySpec, PerformanceMetrics, RiskMetrics
        from src.viz import plot_equity_curve
        from src.tracking import ExperimentManager
        
        print("✅ All modules imported successfully")
        
        # STEP 2: Create a simple strategy
        print("\n📊 Step 2: Creating a simple strategy...")
        
        strategy = StrategySpec(
            name="My First Strategy",
            description="A simple mean reversion strategy",
            version="1.0.0",
            universe=["AAPL", "GOOGL", "MSFT"],
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            features=[
                {
                    "name": "price_deviation",
                    "feature_type": "price_based",
                    "lookback_period": 20,
                    "parameters": {"method": "z_score"}
                }
            ],
            target="next_day_return",
            target_lookback=1,
            entry_rules={
                "condition": "price_deviation < -1.5",
                "lookback_period": 5
            },
            exit_rules={
                "condition": "price_deviation > 0.5",
                "lookback_period": 3
            },
            holding_period=10,
            position_sizing={
                "method": "fixed_size",
                "max_position_size": 0.2
            },
            max_positions=2,
            cv_config={
                "validation_type": "walk_forward",
                "n_splits": 3,
                "train_size": 0.7
            }
        )
        
        print(f"✅ Strategy created: {strategy.name}")
        print(f"   Universe: {strategy.universe}")
        print(f"   Features: {len(strategy.features)}")
        
        # STEP 3: Generate sample data (in real usage, you'd load actual market data)
        print("\n📈 Step 3: Generating sample data...")
        
        # Create 1 year of daily returns
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Simulate strategy returns (in real usage, this comes from your backtest)
        strategy_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
        
        print(f"✅ Generated {len(dates)} days of data")
        
        # STEP 4: Calculate performance metrics
        print("\n🔍 Step 4: Calculating performance metrics...")
        
        # Performance metrics
        perf_calc = PerformanceMetrics()
        performance = perf_calc.calculate_all_metrics(strategy_returns)
        
        # Risk metrics
        risk_calc = RiskMetrics()
        risk = risk_calc.calculate_all_risk_metrics(strategy_returns)
        
        print("\n📊 Key Performance Metrics:")
        print(f"   Annualized Return: {performance.get('annualized_return', 0):.4f}")
        print(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.4f}")
        print(f"   Sortino Ratio: {performance.get('sortino_ratio', 0):.4f}")
        print(f"   Max Drawdown: {risk.get('max_drawdown', 0):.4f}")
        print(f"   Volatility: {risk.get('volatility', 0):.4f}")
        
        # STEP 5: Create visualizations
        print("\n🎨 Step 5: Creating visualizations...")
        
        # Create equity curve
        equity_curve = (1 + strategy_returns).cumprod() * 100000  # Start with $100k
        benchmark_curve = (1 + benchmark_returns).cumprod() * 100000
        
        # Generate simple plot
        fig1 = plot_equity_curve(equity_curve, benchmark_curve, title="My Strategy vs Benchmark")
        print("✅ Equity curve plot created")
        
        # STEP 6: Track your experiment
        print("\n🔬 Step 6: Tracking experiment...")
        
        exp_manager = ExperimentManager(
            experiment_name="my_first_strategy",
            base_output_dir="my_experiments",
            auto_track=False
        )
        
        exp_id = exp_manager.start_strategy_experiment(
            strategy,
            description="My first quant strategy",
            tags={"strategy_type": "mean_reversion", "experience": "beginner"}
        )
        
        # Log your results
        exp_manager.log_experiment_step(
            "strategy_creation",
            {"strategy_name": strategy.name, "universe_size": len(strategy.universe)},
            step_type="info"
        )
        
        exp_manager.log_experiment_step(
            "performance_results",
            {
                "sharpe_ratio": performance.get('sharpe_ratio', 0),
                "max_drawdown": risk.get('max_drawdown', 0),
                "annualized_return": performance.get('annualized_return', 0)
            },
            step_type="metrics"
        )
        
        exp_manager.end_experiment("completed")
        print(f"✅ Experiment tracked with ID: {exp_id}")
        
        # STEP 7: Summary and next steps
        print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n📚 What you just accomplished:")
        print("1. ✅ Created a StrategySpec with features and rules")
        print("2. ✅ Generated sample data (replace with real market data)")
        print("3. ✅ Calculated performance and risk metrics")
        print("4. ✅ Created professional visualizations")
        print("5. ✅ Tracked your experiment")
        
        print("\n🚀 Next steps for real usage:")
        print("1. Replace sample data with real market data (yfinance, Alpha Vantage)")
        print("2. Implement actual feature engineering (technical indicators, fundamentals)")
        print("3. Train ML models on your features")
        print("4. Run comprehensive backtests with the BacktestEngine")
        print("5. Use the AI tools to generate more sophisticated strategies")
        
        print("\n💡 Pro Tips:")
        print("- Start with simple strategies (1-2 features)")
        print("- Always use walk-forward validation")
        print("- Include transaction costs in backtests")
        print("- Track everything in MLflow")
        print("- Generate research reports for each strategy")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
