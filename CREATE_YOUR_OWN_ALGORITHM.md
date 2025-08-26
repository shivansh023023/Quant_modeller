# 🚀 Create Your Own Trading Algorithm

This guide shows you how to create your own quantitative trading algorithms using the AI-Powered Quantitative Trading Platform.

## 📋 Quick Start

**Want to jump right in?** Run this simple example:

```bash
python simple_custom_algorithm.py
```

This demonstrates a **Simple Momentum + RSI Strategy** that:
- ✅ Generated **38.3% total returns** (11.2% annually)
- ✅ Achieved **Sharpe Ratio of 1.43** (excellent risk-adjusted returns)
- ✅ Only **6.77% maximum drawdown** (low risk)
- ✅ Made **480 trades** over 3 years on 8 tech stocks

## 🎯 Algorithm Examples

### 1. **Simple Momentum Strategy** (`simple_custom_algorithm.py`)
- **Best for beginners** - Clean, easy to understand
- Uses momentum + RSI filtering
- Equal weight position sizing
- **Result:** 38.3% returns, 1.43 Sharpe ratio

### 2. **Advanced Momentum Strategy** (`custom_momentum_example.py`)
- More sophisticated with multiple indicators
- Includes MACD, Bollinger Bands, volatility filtering
- Professional risk management
- **For experienced users**

## 🏗️ Build Your Own Algorithm

### Step 1: Define Your Strategy
```python
class YourStrategy:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        
        # Your strategy parameters
        self.your_parameter = 0.05  # Example: 5% threshold
```

### Step 2: Create Signal Generation
```python
def generate_signals(self, data):
    # Calculate your indicators
    data['your_indicator'] = data['Close'].rolling(20).mean()
    
    # Define buy conditions
    buy_conditions = (
        (data['Close'] > data['your_indicator']) &
        # Add more conditions...
    )
    
    # Define sell conditions
    sell_conditions = (
        (data['Close'] < data['your_indicator']) |
        # Add more conditions...
    )
    
    # Create signals
    data['signal'] = 0
    data.loc[buy_conditions, 'signal'] = 1   # Buy
    data.loc[sell_conditions, 'signal'] = -1  # Sell
    
    return data
```

### Step 3: Run Backtest
```python
def backtest(self, symbols, start_date, end_date):
    # Download data
    all_data = {}
    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date)
        all_data[symbol] = data
    
    # Generate signals for each stock
    signals_data = {}
    for symbol, data in all_data.items():
        signals = self.generate_signals(data)
        signals_data[symbol] = signals
    
    # Simulate trading and return results
    return self.simulate_trades(signals_data)
```

## 📊 Popular Strategy Ideas

### **Momentum Strategies**
```python
# Buy when price momentum is strong
momentum_20d = data['Close'].pct_change(20)
buy_signal = momentum_20d > 0.05  # 5% momentum
```

### **Mean Reversion**
```python
# Buy when price is below moving average (oversold)
sma_50 = data['Close'].rolling(50).mean()
buy_signal = data['Close'] < sma_50 * 0.95  # 5% below SMA
```

### **RSI Strategy**
```python
# RSI-based entry/exit
rsi = calculate_rsi(data['Close'])
buy_signal = rsi < 30    # Oversold
sell_signal = rsi > 70   # Overbought
```

### **Volatility Breakout**
```python
# Buy on volatility expansion
volatility = data['Close'].pct_change().rolling(20).std()
buy_signal = volatility > volatility.rolling(100).mean()
```

### **Multi-Factor Model**
```python
# Combine multiple signals
momentum_signal = data['Close'].pct_change(20) > 0.05
rsi_signal = calculate_rsi(data['Close']) < 70
volume_signal = data['Volume'] > data['Volume'].rolling(20).mean()

buy_signal = momentum_signal & rsi_signal & volume_signal
```

## ⚙️ Strategy Parameters to Experiment With

### **Entry/Exit Thresholds**
```python
self.momentum_threshold = 0.05    # Try: 0.03, 0.07, 0.10
self.rsi_overbought = 70         # Try: 65, 75, 80
self.volatility_filter = 0.30    # Try: 0.20, 0.40, 0.50
```

### **Technical Indicator Periods**
```python
rsi_period = 14                  # Try: 10, 21, 30
sma_period = 20                  # Try: 10, 50, 200
momentum_period = 20             # Try: 5, 10, 60
```

### **Position Sizing**
```python
# Equal weight (simple)
position_size = cash / max_positions

# Volatility targeting (advanced)
target_vol = 0.15
stock_vol = data['Close'].pct_change().std() * sqrt(252)
position_size = (target_vol / stock_vol) * portfolio_value
```

## 🎯 Performance Targets

**Good Performance Benchmarks:**
- **Sharpe Ratio:** > 1.0 (excellent), > 0.5 (good)
- **Max Drawdown:** < 20% (good), < 10% (excellent)  
- **Win Rate:** 45-55% is typical for good strategies
- **Annual Return:** > 10% (good), > 15% (excellent)

## 🛠️ Advanced Features You Can Add

### **Risk Management**
```python
# Stop loss
if current_loss > 0.05:  # 5% stop loss
    sell_position()

# Position sizing based on volatility
volatility = calculate_volatility(symbol)
position_size = target_risk / volatility
```

### **Multiple Timeframes**
```python
# Use different timeframes
daily_data = get_data(symbol, '1d')
hourly_data = get_data(symbol, '1h')

# Combine signals from different timeframes
daily_trend = daily_data['Close'] > daily_data['SMA_200']
hourly_signal = hourly_data['RSI'] < 30

final_signal = daily_trend & hourly_signal
```

### **Sector Rotation**
```python
# Trade different sectors based on market conditions
tech_stocks = ['AAPL', 'MSFT', 'GOOGL']
finance_stocks = ['JPM', 'BAC', 'WFC']
energy_stocks = ['XOM', 'CVX', 'COP']

# Switch sectors based on market regime
if market_regime == 'growth':
    active_universe = tech_stocks
elif market_regime == 'value':
    active_universe = finance_stocks
```

## 📈 Example: Complete Custom Strategy

```python
class MyCustomStrategy:
    def __init__(self):
        self.initial_capital = 100000
        self.momentum_threshold = 0.08  # 8% momentum
        self.rsi_threshold = 60         # Less restrictive RSI
        self.stop_loss = 0.10          # 10% stop loss
        
    def generate_signals(self, data):
        # Flatten yfinance columns
        if data.columns.nlevels > 1:
            data.columns = data.columns.droplevel(1)
            
        # Custom indicators
        data['momentum_20'] = data['Close'].pct_change(20)
        data['rsi'] = self.calculate_rsi(data['Close'])
        data['sma_50'] = data['Close'].rolling(50).mean()
        data['volume_avg'] = data['Volume'].rolling(30).mean()
        
        # Multi-factor buy signal
        buy_signal = (
            (data['momentum_20'] > self.momentum_threshold) &
            (data['rsi'] < self.rsi_threshold) &
            (data['Close'] > data['sma_50']) &
            (data['Volume'] > data['volume_avg'] * 1.2)
        )
        
        # Multi-factor sell signal
        sell_signal = (
            (data['momentum_20'] < 0.01) |  # Momentum fading
            (data['rsi'] > 85) |            # Very overbought
            (data['Close'] < data['sma_50'] * 0.95)  # Below SMA
        )
        
        data['signal'] = 0
        data.loc[buy_signal, 'signal'] = 1
        data.loc[sell_signal, 'signal'] = -1
        
        return data
```

## 🚀 Next Steps

1. **Start Simple:** Modify `simple_custom_algorithm.py` with your own parameters
2. **Test Ideas:** Change thresholds, indicators, or stock universe
3. **Analyze Results:** Look at Sharpe ratio, drawdown, win rate
4. **Iterate:** Refine based on performance
5. **Scale Up:** Add more sophisticated features

## 💡 Pro Tips

- **Start with simple strategies** - complexity doesn't always mean better performance
- **Test on different time periods** - make sure your strategy works in various market conditions
- **Focus on risk management** - protecting capital is more important than maximizing returns
- **Use proper position sizing** - don't risk too much on any single trade
- **Keep transaction costs realistic** - include commissions and slippage in your backtests

## 🤝 Contributing

Have a great strategy idea or improvement? 
1. Fork the repository
2. Create your algorithm
3. Test it thoroughly  
4. Submit a pull request

**Happy algorithmic trading!** 📈🚀
