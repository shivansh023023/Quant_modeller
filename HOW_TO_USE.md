# 🚀 How to Use Quant Lab - Complete User Guide

## 🎯 **What You Can Do With Quant Lab**

Quant Lab is an AI-powered quantitative trading strategy development tool that helps you:
- **Create trading strategies** from natural language ideas
- **Test strategies** with realistic backtesting
- **Analyze performance** with professional metrics
- **Track experiments** with MLflow integration
- **Generate reports** and visualizations automatically

## 🚀 **Quick Start (5 Minutes)**

### **Step 1: Test Your Setup**
```bash
# Make sure you're in the quant-lab directory
cd /path/to/quant-lab

# Test basic functionality
python demo.py

# Run the simple workflow example
python simple_example.py
```

### **Step 2: Verify Everything Works**
```bash
# Check configuration
python test_config.py
```

## 📚 **Complete User Workflow**

### **Phase 1: Strategy Creation**

#### **1.1 Create a Strategy Specification**
```python
from src import StrategySpec
from datetime import datetime, timedelta

# Define your trading idea
strategy = StrategySpec(
    name="My Mean Reversion Strategy",
    description="Buy oversold stocks, sell overbought ones",
    universe=["AAPL", "GOOGL", "MSFT", "AMZN"],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    features=[
        {
            "name": "price_zscore",
            "feature_type": "price_based",
            "lookback_period": 20,
            "parameters": {"method": "z_score"}
        }
    ],
    target="next_day_return",
    entry_rules={"condition": "price_zscore < -2.0"},
    exit_rules={"condition": "price_zscore > 0.5"},
    position_sizing={"method": "fixed_size", "max_position_size": 0.2}
)
```

#### **1.2 Define Your Features**
```python
# Price-based features
{
    "name": "momentum_20d",
    "feature_type": "price_based",
    "lookback_period": 20,
    "parameters": {"method": "returns"}
}

# Volume features
{
    "name": "volume_ratio",
    "feature_type": "volume_based", 
    "lookback_period": 10,
    "parameters": {"method": "volume_ratio"}
}

# Volatility features
{
    "name": "volatility_30d",
    "feature_type": "volatility",
    "lookback_period": 30,
    "parameters": {"method": "rolling_std"}
}
```

### **Phase 2: Data & Feature Engineering**

#### **2.1 Load Market Data**
```python
# Option 1: Use yfinance (free)
import yfinance as yf

# Download data for your universe
data = {}
for ticker in strategy.universe:
    ticker_data = yf.download(ticker, start=strategy.start_date, end=strategy.end_date)
    data[ticker] = ticker_data

# Option 2: Use Alpha Vantage (premium)
from src.core.data_api import DataAPI
data_api = DataAPI()
data = data_api.get_ohlcv_data(strategy.universe, strategy.start_date, strategy.end_date)

# Option 3: Load from CSV
import pandas as pd
data = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)
```

#### **2.2 Generate Features**
```python
from src.features.generators import FeatureGenerator

# Create feature generator
feature_gen = FeatureGenerator()

# Generate features for your strategy
features_df = feature_gen.generate_features(
    data, 
    strategy.features,
    target_col="returns"
)
```

### **Phase 3: Model Training & Validation**

#### **3.1 Train Your Model**
```python
from src.models.model_zoo import ModelZoo

# Get available models
model_zoo = ModelZoo()

# Choose a model (e.g., Random Forest)
model = model_zoo.get_model("random_forest")

# Train the model
model.fit(features_df.drop("target", axis=1), features_df["target"])

# Make predictions
predictions = model.predict(features_df.drop("target", axis=1))
```

#### **3.2 Validate Your Strategy**
```python
from src.core.cross_validation import CrossValidator

# Create validator
validator = CrossValidator(
    validation_type="walk_forward",
    n_splits=5,
    train_size=0.7
)

# Run validation
cv_results = validator.cross_validate(
    model, 
    features_df.drop("target", axis=1), 
    features_df["target"]
)
```

### **Phase 4: Backtesting**

#### **4.1 Run Backtest**
```python
from src.backtest import BacktestEngine

# Create backtest engine
backtest = BacktestEngine(
    strategy=strategy,
    data=data,
    initial_capital=100000,
    transaction_costs=0.001,
    slippage_model="fixed"
)

# Run backtest
results = backtest.run()

# Get key results
equity_curve = results.equity_curve
trades = results.trades
metrics = results.metrics
```

### **Phase 5: Analysis & Reporting**

#### **5.1 Calculate Performance Metrics**
```python
from src import PerformanceMetrics, RiskMetrics

# Performance metrics
perf_calc = PerformanceMetrics()
performance = perf_calc.calculate_all_metrics(returns)

# Risk metrics  
risk_calc = RiskMetrics()
risk = risk_calc.calculate_all_risk_metrics(returns)

print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {risk['max_drawdown']:.3f}")
```

#### **5.2 Create Visualizations**
```python
from src.viz import plot_equity_curve, plot_drawdown, plot_monthly_returns

# Equity curve
fig1 = plot_equity_curve(equity_curve, title="Strategy Performance")

# Drawdown analysis
fig2 = plot_drawdown(equity_curve, title="Strategy Drawdown")

# Monthly returns heatmap
fig3 = plot_monthly_returns(returns, title="Monthly Returns")
```

#### **5.3 Generate Research Report**
```python
from src.viz import generate_performance_report

# Generate comprehensive report
report_path = generate_performance_report(
    returns, 
    strategy_name="My Strategy",
    output_dir="reports"
)
```

### **Phase 6: Experiment Tracking**

#### **6.1 Track Your Experiment**
```python
from src.tracking import ExperimentManager

# Start tracking
exp_manager = ExperimentManager("my_strategy_experiment")
exp_id = exp_manager.start_strategy_experiment(strategy)

# Log key steps
exp_manager.log_experiment_step(
    "feature_generation",
    {"features_created": len(strategy.features)},
    step_type="info"
)

exp_manager.log_experiment_step(
    "model_training",
    {"accuracy": 0.75, "training_time": 2.1},
    step_type="metrics"
)

exp_manager.log_experiment_step(
    "backtest_results",
    {
        "sharpe_ratio": performance['sharpe_ratio'],
        "max_drawdown": risk['max_drawdown'],
        "annualized_return": performance['annualized_return']
    },
    step_type="metrics"
)

# End experiment
exp_manager.end_experiment("completed")
```

## 🎯 **Common Strategy Examples**

### **Example 1: Mean Reversion Strategy**
```python
strategy = StrategySpec(
    name="Mean Reversion Strategy",
    features=[
        {
            "name": "price_zscore",
            "feature_type": "price_based",
            "lookback_period": 20,
            "parameters": {"method": "z_score"}
        }
    ],
    entry_rules={"condition": "price_zscore < -2.0"},
    exit_rules={"condition": "price_zscore > 0.5"},
    holding_period=5
)
```

### **Example 2: Momentum Strategy**
```python
strategy = StrategySpec(
    name="Momentum Strategy", 
    features=[
        {
            "name": "price_momentum",
            "feature_type": "price_based",
            "lookback_period": 20,
            "parameters": {"method": "returns"}
        },
        {
            "name": "volume_momentum",
            "feature_type": "volume_based",
            "lookback_period": 10,
            "parameters": {"method": "volume_ratio"}
        }
    ],
    entry_rules={"condition": "price_momentum > 0.05 AND volume_momentum > 1.2"},
    exit_rules={"condition": "price_momentum < -0.02"},
    holding_period=10
)
```

### **Example 3: Multi-Factor Strategy**
```python
strategy = StrategySpec(
    name="Multi-Factor Strategy",
    features=[
        {
            "name": "value_score",
            "feature_type": "fundamental",
            "lookback_period": 60,
            "parameters": {"method": "pe_ratio"}
        },
        {
            "name": "momentum_score", 
            "feature_type": "price_based",
            "lookback_period": 20,
            "parameters": {"method": "returns"}
        },
        {
            "name": "quality_score",
            "feature_type": "fundamental", 
            "lookback_period": 60,
            "parameters": {"method": "roe"}
        }
    ],
    entry_rules={"condition": "value_score > 0.7 AND momentum_score > 0.6 AND quality_score > 0.8"},
    exit_rules={"condition": "value_score < 0.3 OR momentum_score < 0.2"},
    holding_period=20
)
```

## 🔧 **Configuration & Setup**

### **API Keys Setup**
1. Edit `configs/api_keys.yml`
2. Add your keys:
   ```yaml
   alpha_vantage:
     api_key: "your_key_here"
   gemini:
     api_key: "your_key_here"
   ```
3. Test with `python test_config.py`

### **Data Sources**
- **Yahoo Finance**: Free, good for testing
- **Alpha Vantage**: Premium data, better quality  
- **CSV Files**: Your own data
- **Custom APIs**: Integrate your data sources

## 📊 **Understanding Results**

### **Key Performance Metrics**
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

## 🚨 **Best Practices**

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

## 🎓 **Learning Path**

### **Beginner (Week 1-2)**
1. Run `demo.py` and `simple_example.py`
2. Study the code examples
3. Create a simple strategy with 1-2 features
4. Test with sample data

### **Intermediate (Week 3-8)**
1. Load real market data (yfinance)
2. Implement actual feature engineering
3. Train ML models (Random Forest, XGBoost)
4. Run comprehensive backtests

### **Advanced (Month 3+)**
1. Build custom slippage models
2. Implement advanced position sizing
3. Create ensemble strategies
4. Optimize hyperparameters with Optuna

## 🆘 **Troubleshooting**

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
- Check error messages carefully
- Verify your Python environment
- Ensure all dependencies are installed
- Test with demo scripts first

## 📁 **File Organization**

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
├── simple_example.py      # Simple workflow
└── example_momentum_strategy.py  # Strategy example
```

## 🎉 **You're Ready to Start!**

### **First Steps**
1. **Test the system**: Run `python demo.py`
2. **Learn the workflow**: Study `simple_example.py`
3. **Create your strategy**: Start with a simple idea
4. **Test with data**: Use yfinance for free data
5. **Iterate and improve**: Track everything in MLflow

### **Remember**
- Start simple
- Test thoroughly  
- Validate properly
- Track everything
- Learn from each experiment

**Happy trading! 📈**

---

## 📞 **Need Help?**

If you run into issues:
1. Check the error messages
2. Verify your setup with `python test_config.py`
3. Test with the demo scripts
4. Review the best practices section
5. Start with simple examples and build up

The system is designed to be user-friendly, so if something seems complicated, there's probably a simpler way to do it!
