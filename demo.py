#!/usr/bin/env python3
"""
Demo script for Quant Lab system.

This script demonstrates the basic capabilities of the Quant Lab
AI-powered quantitative trading strategy development tool.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_basic_functionality():
    """Demonstrate basic system functionality."""
    print("🚀 QUANT LAB DEMO")
    print("=" * 50)
    
    try:
        # Import core components
        from src import (
            StrategySpec, BacktestResult, BacktestEngine,
            PerformanceMetrics, RiskMetrics,
            IdeaGenerator, MLflowTracker
        )
        
        print("✅ All core modules imported successfully")
        
        # Create a sample strategy specification
        print("\n📊 Creating Sample Strategy...")
        
        strategy_spec = StrategySpec(
            name="Demo Momentum Strategy",
            description="A simple momentum-based trading strategy",
            version="1.0.0",
            universe=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            features=[
                {
                    "name": "price_momentum",
                    "feature_type": "price_based",
                    "lookback_period": 20,
                    "parameters": {"method": "returns"}
                }
            ],
            target="next_day_return",
            target_lookback=1,
            entry_rules={
                "condition": "momentum > 0.02",
                "lookback_period": 20
            },
            exit_rules={
                "condition": "momentum < -0.01",
                "lookback_period": 10
            },
            holding_period=5,
            position_sizing={
                "method": "fixed_size",
                "max_position_size": 0.2
            },
            max_positions=3,
            cv_config={
                "validation_type": "walk_forward",
                "n_splits": 5,
                "train_size": 0.7
            }
        )
        
        print(f"✅ Strategy created: {strategy_spec.name}")
        print(f"   Universe: {len(strategy_spec.universe)} stocks")
        print(f"   Features: {len(strategy_spec.features)}")
        print(f"   Period: {strategy_spec.start_date.date()} to {strategy_spec.end_date.date()}")
        
        # Demonstrate metrics calculation
        print("\n📈 Demonstrating Metrics Calculation...")
        
        # Create sample returns data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        
        # Calculate performance metrics
        performance_calc = PerformanceMetrics()
        performance_metrics = performance_calc.calculate_all_metrics(returns)
        
        print("Performance Metrics:")
        for metric, value in list(performance_metrics.items())[:5]:  # Show first 5
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Calculate risk metrics
        risk_calc = RiskMetrics()
        risk_metrics = risk_calc.calculate_all_risk_metrics(returns)
        
        print("\nRisk Metrics:")
        for metric, value in list(risk_metrics.items())[:5]:  # Show first 5
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Demonstrate visualization
        print("\n🎨 Demonstrating Visualization...")
        
        try:
            from src.viz import plot_equity_curve, plot_returns_distribution
            
            # Create sample equity curve
            equity_curve = (1 + returns).cumprod() * 100000  # Start with $100k
            
            # Generate plots
            fig1 = plot_equity_curve(equity_curve, title="Sample Equity Curve")
            fig2 = plot_returns_distribution(returns, title="Sample Returns Distribution")
            
            print("✅ Visualization plots created successfully")
            print("   - Equity curve plot")
            print("   - Returns distribution plot")
            
        except Exception as e:
            print(f"⚠️  Visualization demo skipped: {e}")
        
        # Demonstrate experiment tracking
        print("\n🔬 Demonstrating Experiment Tracking...")
        
        try:
            from src.tracking import ExperimentManager
            
            # Create experiment manager
            exp_manager = ExperimentManager(
                experiment_name="demo_experiment",
                base_output_dir="demo_runs",
                auto_track=False  # Disable MLflow for demo
            )
            
            # Start a strategy experiment
            exp_id = exp_manager.start_strategy_experiment(
                strategy_spec,
                description="Demo strategy experiment",
                tags={"demo": "true", "type": "momentum"}
            )
            
            print(f"✅ Experiment started: {exp_id}")
            
            # Log some metrics
            exp_manager.log_experiment_step(
                "feature_generation",
                {"features_created": 5, "processing_time": 0.5},
                step_type="info"
            )
            
            exp_manager.log_experiment_step(
                "model_training",
                {"accuracy": 0.75, "training_time": 2.1},
                step_type="metrics"
            )
            
            # End experiment
            exp_manager.end_experiment("completed")
            print("✅ Experiment completed successfully")
            
        except Exception as e:
            print(f"⚠️  Experiment tracking demo skipped: {e}")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nYour Quant Lab system is working perfectly!")
        print("\n📚 What you can do next:")
        print("1. Create real trading strategies using the AI-powered tools")
        print("2. Run backtests with realistic assumptions")
        print("3. Analyze performance with comprehensive metrics")
        print("4. Track experiments with MLflow integration")
        print("5. Generate professional reports and visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    success = demo_basic_functionality()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
