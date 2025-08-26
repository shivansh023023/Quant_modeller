#  Quant Modeller - AI-Powered Quantitative Trading Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-83%20Passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-32%25-yellow)](htmlcov/index.html)

> **A comprehensive research workbench for systematic trading strategies, inspired by Two Sigma's quantitative research workflow**

##  Overview

**Quant Modeller** is a sophisticated, AI-powered quantitative trading strategy development platform that transforms natural language trading ideas into robust, validated strategies with professional-grade analytics and risk management.

###  Key Features

-  **AI-Powered Strategy Generation** - Convert natural language ideas to structured strategies
-  **Professional Analytics** - 40+ performance and risk metrics
-  **Rigorous Validation** - Walk-forward analysis with purged cross-validation
-  **Model Zoo** - 23+ ML models including ensembles and deep learning
-  **Advanced Backtesting** - Realistic transaction costs and slippage modeling
-  **Interactive Dashboards** - Professional visualization suite
-  **Experiment Tracking** - MLflow integration for reproducible research
-  **Auto-Generated Reports** - AI-powered research note generation

##  What Makes This Special

- **Production-Ready**: Enterprise-grade code quality with comprehensive testing
- **Institutional-Quality**: Features used by top-tier hedge funds and asset managers
- **AI-Integrated**: Natural language to strategy pipeline with explainable AI
- **Data Leakage Prevention**: Built-in temporal validation and purging mechanisms
- **Comprehensive**: End-to-end workflow from idea generation to deployment

##  Demo

```bash
# Quick start demo
python demo.py
```

*Sample output showing strategy performance metrics and visualizations*

##  Architecture

```
Quant Modeller/
├── src/
│   ├── strategies/         # Strategy specification and validation
│   ├── ai/                 # AI-powered idea generation & explainability
│   ├── features/          # Feature engineering (100+ financial features)
│   ├── models/            # ML model zoo (23+ algorithms)
│   ├── backtest/          # Backtesting engine with realistic assumptions
│   ├── core/              # Cross-validation, metrics, data management
│   ├── tracking/          # MLflow integration & experiment management
│   └── viz/               # Interactive dashboards & reporting
├── tests/                 # Comprehensive test suite (83 tests)
├── configs/               # Configuration and API key management
└── data/                  # Data storage with automated cataloging
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shivansh023023/Quant_modeller.git
cd Quant_modeller

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 2. Set Up API Keys

**Required API Keys:**
- **Alpha Vantage**: Market data and fundamental data
- **Gemini AI**: AI-powered strategy generation and research notes

**Get Free API Keys:**
- Alpha Vantage: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- Gemini AI: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

**Setup Options:**

**Option A: Interactive Setup (Recommended)**
```bash
python setup_api_keys.py
```

**Option B: Manual Configuration**
1. Copy `configs/api_keys.yml` to `configs/api_keys_local.yml`
2. Edit `configs/api_keys_local.yml` with your actual API keys
3. Never commit your actual API keys to version control!

**Option C: Environment Variables**
```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
```

### 3. Test Configuration

```python
from src.core.config import config_manager

# Check configuration status
config_manager.print_config_summary()
```

### 4. Basic Usage

```python
from src import StrategySpec, PerformanceMetrics
from datetime import datetime, timedelta

# Create a momentum strategy
strategy = StrategySpec(
    name="Momentum Strategy",
    description="Buy strong momentum stocks with high volume",
    universe=["AAPL", "GOOGL", "MSFT", "AMZN"],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    features=[{
        "name": "momentum_20",
        "feature_type": "price_based",
        "lookback_period": 20,
        "parameters": {"method": "returns"}
    }],
    target="next_day_return",
    entry_rules={"condition": "momentum_20 > 0.05"},
    exit_rules={"condition": "momentum_20 < -0.02"},
    holding_period=10
)

# Analyze performance
metrics = PerformanceMetrics()
# Your strategy analysis here...
```

## 📚 Complete User Guide

### 1. Strategy Development Workflow

```python
# Step 1: Define your strategy
from src import StrategySpec

strategy = StrategySpec(
    name="My Trading Strategy",
    universe=["AAPL", "GOOGL", "MSFT"],
    features=[...],
    entry_rules={...},
    exit_rules={...}
)

# Step 2: Generate features automatically
from src.features import FeatureGenerator

feature_gen = FeatureGenerator()
features = feature_gen.generate_all_features(market_data)

# Step 3: Train ML models
from src.models import ModelZoo

zoo = ModelZoo()
model = zoo.get_model("random_forest")
model.fit(features, targets)

# Step 4: Backtest with realistic assumptions
from src.backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    transaction_costs=0.001,  # 10 bps
    slippage_bps=5.0
)
results = engine.run_backtest(strategy)

# Step 5: Analyze results
print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results.metrics['max_drawdown']:.2%}")
```

### 2. AI-Powered Strategy Generation

```python
from src.ai import IdeaGenerator

# Convert natural language to strategy
generator = IdeaGenerator()
strategy = generator.generate_strategy(
    "Buy stocks when RSI < 30 and volume > 20-day average, hold for 5 days"
)

# Get AI explanations
explanation = generator.explain_strategy_decision(strategy, date, market_data)
```

### 3. Advanced Analytics

```python
from src import PerformanceMetrics, RiskMetrics
from src.viz import create_strategy_dashboard

# Calculate comprehensive metrics
perf_metrics = PerformanceMetrics()
risk_metrics = RiskMetrics()

performance = perf_metrics.calculate_all_metrics(returns)
risk = risk_metrics.calculate_all_risk_metrics(returns)

# Create interactive dashboard
dashboard = create_strategy_dashboard(
    equity_curve=results.equity_curve,
    returns=results.returns,
    benchmark_returns=benchmark_returns
)
```

### 4. Experiment Tracking

```python
from src.tracking import ExperimentManager

# Track your research
exp_manager = ExperimentManager("my_strategy_research")
exp_id = exp_manager.start_strategy_experiment(strategy)

# Log results automatically
exp_manager.log_experiment_results({
    "sharpe_ratio": 1.25,
    "max_drawdown": -0.15,
    "total_return": 0.23
})
```

## 🔑 Core Concepts

### StrategySpec Schema

The `StrategySpec` is a Pydantic model that defines:
- **Universe**: Target securities and time period
- **Features**: Input variables (with lookback periods)
- **Target**: Prediction target (returns, direction, etc.)
- **Rules**: Entry/exit conditions and holding periods
- **Constraints**: Risk management and position sizing
- **Validation**: Cross-validation setup

### Data Leakage Prevention

- All features use only past information
- Lookback periods are enforced at the schema level
- Walk-forward validation prevents temporal leakage
- Purged cross-validation for training

### Backtesting Assumptions

- Daily bar resolution
- Next-open fills
- Transaction costs and slippage
- Volatility-based position sizing
- Equal-weight portfolio construction

## 📊 Validation Framework

1. **Walk-Forward Analysis**: Time-based splits for final evaluation
2. **Purged Cross-Validation**: Removes overlapping periods during training
3. **Robustness Checks**: Monte Carlo resampling, cost shocks, noise features
4. **Performance Metrics**: Sharpe, Sortino, MaxDD, turnover, hit rate

## 🛠 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## 🔧 Advanced Features

### Machine Learning Models (23 Available)

**Linear Models**: Linear/Logistic Regression, Ridge, Lasso, Elastic Net, SVM
**Tree Models**: Random Forest, Gradient Boosting, Extra Trees, XGBoost
**Neural Networks**: Multi-layer Perceptron with customizable architectures  
**Ensemble Models**: Voting, Bagging, Stacking with automatic optimization

### Feature Engineering (100+ Features)

- **Price Features**: Returns, moving averages, momentum indicators
- **Volume Features**: Volume ratios, volume-weighted prices  
- **Volatility Features**: Realized volatility, Parkinson, Garman-Klass
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic
- **Cross-Sectional**: Universe rankings, relative strength
- **Market Regime**: Volatility regimes, trend detection

### Risk Management

- **Position Sizing**: Volatility targeting, Kelly criterion, fixed size
- **Transaction Costs**: Configurable basis points, market impact modeling
- **Slippage Models**: Linear, square-root, custom implementations
- **Risk Controls**: Stop-loss, take-profit, correlation limits

## 📊 Performance & Validation

### Cross-Validation Methods
- **Walk-Forward Analysis**: Expanding window validation
- **Purged K-Fold**: Removes overlapping periods to prevent leakage
- **Time Series Split**: Respects temporal structure

### Metrics (40+ Available)
- **Performance**: Sharpe, Sortino, Calmar ratios, Win rate
- **Risk**: VaR, CVaR, Maximum Drawdown, Beta, Tracking Error
- **Distribution**: Skewness, Kurtosis, Tail risk measures

## 📈 Use Cases

### For Portfolio Managers
- Multi-factor model development
- Risk-adjusted portfolio optimization
- Regime-aware strategy allocation

### For Quantitative Researchers  
- Academic research with publication-quality results
- Factor discovery and validation
- Cross-asset strategy development

### For Algorithmic Traders
- Systematic strategy development
- Live trading system prototyping
- Performance attribution analysis

### For Financial Technology
- Robo-advisor algorithm development  
- Risk management system design
- Alternative data integration

## 🎓 Learning Path

**Beginner**: Start with `demo.py` and `simple_example.py`
**Intermediate**: Build multi-factor strategies with cross-validation
**Advanced**: Develop ensemble models with custom features
**Expert**: Create production trading systems with real-time data

## 📋 Requirements

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning
- `lightgbm>=4.0.0` - Gradient boosting
- `xgboost>=2.0.0` - Extreme gradient boosting
- `plotly>=5.15.0` - Interactive visualizations
- `mlflow>=2.6.0` - Experiment tracking
- `pydantic>=2.0.0` - Data validation

### Data Sources
- **Yahoo Finance** (free): Basic market data
- **Alpha Vantage** (premium): Enhanced data quality
- **Custom CSV**: Your proprietary datasets

## 📈 Example Strategies

### Mean Reversion
```python
strategy = StrategySpec(
    universe=["SPY", "QQQ", "IWM"],
    features=["rsi_14", "bb_position_20", "volume_ratio_20"],
    target="next_day_return",
    entry_rules="rsi_14 < 30 and bb_position_20 < 0.1",
    exit_rules="rsi_14 > 70 or bb_position_20 > 0.9",
    holding_period=5
)
```

### Momentum
```python
strategy = StrategySpec(
    universe=["SPY", "QQQ", "IWM"],
    features=["momentum_20", "volatility_20", "volume_ratio_20"],
    target="next_day_return",
    entry_rules="momentum_20 > 0.05 and volume_ratio_20 > 1.2",
    exit_rules="momentum_20 < -0.02",
    holding_period=10
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Two Sigma's quantitative research methodologies
- Built with modern software engineering practices
- Designed for both academic research and industry applications

## 📞 Contact

**Shivansh** - [@shivansh023023](https://github.com/shivansh023023)

Project Link: [https://github.com/shivansh023023/Quant_modeller](https://github.com/shivansh023023/Quant_modeller)

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It is not intended for live trading without proper risk management and thorough validation. Always backtest strategies extensively before any real-world application.

---

⭐ **Star this repository if you find it helpful for your quantitative research!**
