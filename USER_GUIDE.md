# 🚀 Quant Lab - User Guide

## Quick Start

### 1. **Test the System**
```bash
# Basic functionality test
python demo.py

# Run the momentum strategy example
python example_momentum_strategy.py
```

### 2. **Check Your Setup**
```bash
# Verify configuration
python test_config.py
```

## 📚 How to Use Quant Lab

### **Step 1: Create a Trading Strategy**

```python
from src import StrategySpec
from datetime import datetime, timedelta

# Define your strategy
strategy = StrategySpec(
    name="My First Strategy",
    description="A simple mean reversion strategy",
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
    entry_rules={"condition": "price_deviation < -1.5"},
    exit_rules={"condition": "price_deviation > 0.5"},
    position_sizing={"method": "fixed_size", "max_position_size": 0.2}
)
```

### **Step 2: Analyze Performance**

```python
from src import PerformanceMetrics, RiskMetrics
import pandas as pd

# Your strategy returns (replace with actual data)
returns = pd.Series([0.01, -0.005, 0.02, ...])

# Calculate metrics
perf_calc = PerformanceMetrics()
performance = perf_calc.calculate_all_metrics(returns)

risk_calc = RiskMetrics()
risk = risk_calc.calculate_all_risk_metrics(returns)

print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {risk['max_drawdown']:.3f}")
```

### **Step 3: Create Visualizations**

```python
from src.viz import plot_equity_curve, create_strategy_dashboard

# Create equity curve
equity = (1 + returns).cumprod() * 100000
fig = plot_equity_curve(equity, title="My Strategy Performance")

# Create comprehensive dashboard
dashboard = create_strategy_dashboard(
    equity, returns, title="Strategy Dashboard"
)
```

### **Step 4: Track Experiments**

```python
from src.tracking import ExperimentManager

# Start tracking
exp_manager = ExperimentManager("my_strategy_experiment")
exp_id = exp_manager.start_strategy_experiment(strategy)

# Log results
exp_manager.log_experiment_step(
    "backtest_completed",
    {"sharpe_ratio": 1.2, "max_drawdown": 0.15}
)

exp_manager.end_experiment("completed")
```

### **Step 5: Generate Reports**

```python
from src.viz import generate_performance_report

# Generate research report
report_path = generate_performance_report(
    returns, strategy_name="My Strategy"
)
print(f"Report saved to: {report_path}")
```

## 🎯 Common Use Cases

### **Case 1: Mean Reversion Strategy**
- **Idea**: Buy oversold stocks, sell overbought ones
- **Features**: Z-scores, Bollinger Bands, RSI
- **Entry**: Z-score < -2 (oversold)
- **Exit**: Z-score > 0 (mean reversion)

### **Case 2: Momentum Strategy**
- **Idea**: Follow the trend
- **Features**: Price momentum, volume momentum, volatility
- **Entry**: Strong positive momentum
- **Exit**: Momentum reversal or high volatility

### **Case 3: Factor Model**
- **Idea**: Multi-factor stock selection
- **Features**: Value, momentum, quality, size
- **Entry**: High composite score
- **Exit**: Score deterioration

## 🔧 Configuration

### **API Keys Setup**
1. Edit `configs/api_keys.yml`
2. Add your Alpha Vantage, Gemini, etc. keys
3. Test with `python test_config.py`

### **Data Sources**
- **Yahoo Finance**: Free, good for testing
- **Alpha Vantage**: Premium data, better quality
- **CSV Files**: Your own data

## 📊 Understanding Results

### **Key Metrics**
- **Sharpe Ratio**: Risk-adjusted returns (>1 is good)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Annual return / Max drawdown
- **Win Rate**: Percentage of profitable trades

### **Risk Metrics**
- **VaR (95%)**: 95% confidence loss limit
- **CVaR**: Expected loss beyond VaR
- **Volatility**: Standard deviation of returns
- **Beta**: Market correlation

## 🚨 Best Practices

### **Data Leakage Prevention**
- ✅ Use only past data for features
- ✅ Implement walk-forward validation
- ✅ Use purged cross-validation
- ❌ Never use future prices in features

### **Backtesting**
- ✅ Include transaction costs
- ✅ Model slippage realistically
- ✅ Use proper position sizing
- ❌ Don't overfit to historical data

### **Validation**
- ✅ Test on out-of-sample data
- ✅ Use multiple time periods
- ✅ Check robustness with noise
- ❌ Don't cherry-pick results

## 🆘 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd /path/to/quant-lab
   python -c "import src; print('OK')"
   ```

2. **Configuration Issues**
   ```bash
   # Test configuration
   python test_config.py
   ```

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### **Getting Help**
- Check the error messages carefully
- Verify your Python environment
- Ensure all dependencies are installed
- Test with the demo scripts first

## 🎓 Learning Path

### **Beginner**
1. Run `demo.py` to see the system work
2. Study `example_momentum_strategy.py`
3. Create a simple strategy with 1-2 features
4. Test with sample data

### **Intermediate**
1. Implement real feature engineering
2. Use actual market data
3. Train ML models
4. Run comprehensive backtests

### **Advanced**
1. Build custom slippage models
2. Implement advanced position sizing
3. Create ensemble strategies
4. Optimize hyperparameters

## 📁 File Structure

```
quant-lab/
├── src/                    # Main source code
│   ├── core/              # Core functionality
│   ├── strategies/        # Strategy definitions
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── backtest/          # Backtesting engine
│   ├── tracking/          # Experiment tracking
│   └── viz/               # Visualization
├── configs/               # Configuration files
├── data/                  # Data storage
├── runs/                  # Experiment outputs
├── reports/               # Generated reports
├── demo.py                # Basic demo
└── example_momentum_strategy.py  # Strategy example
```

## 🎉 You're Ready!

Start with the demo scripts, then build your own strategies. Remember:
- Start simple
- Test thoroughly
- Validate properly
- Track everything

Happy trading! 📈
