"""
Prompt templates for LLM-based strategy generation.

This module provides structured prompts that guide LLMs to generate
valid StrategySpec JSONs while preventing data leakage.
"""

from typing import Dict, List, Any
from ..strategies.schema import FeatureType, TargetType, ValidationType


class PromptTemplates:
    """Templates for LLM prompts in strategy generation."""
    
    @staticmethod
    def get_strategy_generation_prompt(
        natural_language_idea: str,
        available_features: List[str] = None,
        universe_constraints: List[str] = None
    ) -> str:
        """
        Generate the main prompt for converting natural language to StrategySpec.
        
        Args:
            natural_language_idea: Natural language description of the strategy
            available_features: List of available feature names
            universe_constraints: Constraints on universe selection
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""
You are an expert quantitative trading strategist. Convert the following natural language trading idea into a structured StrategySpec JSON.

TRADING IDEA: {natural_language_idea}

CRITICAL REQUIREMENTS:
1. NO DATA LEAKAGE: All features must use only past information
2. Lookback periods must be reasonable (typically 5-252 days)
3. Entry/exit rules must not reference future prices or data
4. Strategy period must be at least 1 year (252 trading days)

AVAILABLE FEATURES: {available_features or "Any technical/fundamental features"}

UNIVERSE CONSTRAINTS: {universe_constraints or "US equities, ETFs, or indices"}

Generate a complete StrategySpec JSON with the following structure:
- name: Descriptive strategy name
- description: Clear explanation of the strategy logic
- universe: List of 3-10 ticker symbols
- start_date: "2020-01-01T00:00:00"
- end_date: "2024-01-01T00:00:00"
- features: List of FeatureSpec objects with appropriate lookback periods
- target: One of {[t.value for t in TargetType]}
- target_lookback: 1
- entry_rules: EntryExitRule with condition and lookback
- exit_rules: EntryExitRule with condition and lookback
- holding_period: 5-20 days
- position_sizing: PositionSizing with method and parameters
- max_positions: 5-20
- cv_config: CrossValidationConfig with walk_forward validation

Ensure all lookback periods are consistent and prevent data leakage.
"""
        return base_prompt
    
    @staticmethod
    def get_feature_suggestion_prompt(
        strategy_description: str,
        existing_features: List[str] = None
    ) -> str:
        """
        Generate prompt for suggesting additional features.
        
        Args:
            strategy_description: Description of the strategy
            existing_features: List of already selected features
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a quantitative researcher suggesting features for a trading strategy.

STRATEGY: {strategy_description}

EXISTING FEATURES: {existing_features or "None"}

Suggest 5-10 additional features that would complement this strategy:
- Focus on features that capture different market dynamics
- Ensure lookback periods are appropriate (5-252 days)
- Mix price-based, volume-based, and technical indicators
- Avoid features that are highly correlated with existing ones

For each feature, provide:
1. Feature name (e.g., "rsi_14", "volume_ratio_20")
2. Feature type (price_based, volume_based, technical_indicator, fundamental)
3. Lookback period (days)
4. Brief explanation of why it's relevant

Format as a JSON list of FeatureSpec objects.
"""
        return prompt
    
    @staticmethod
    def get_risk_management_prompt(
        strategy_description: str,
        current_risk_params: Dict[str, Any] = None
    ) -> str:
        """
        Generate prompt for suggesting risk management parameters.
        
        Args:
            strategy_description: Description of the strategy
            current_risk_params: Current risk parameters if any
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a risk management expert for quantitative trading strategies.

STRATEGY: {strategy_description}

CURRENT RISK PARAMS: {current_risk_params or "None"}

Suggest appropriate risk management parameters:
1. Position sizing method (volatility_target, risk_per_trade, fixed_size)
2. Stop loss threshold (if applicable)
3. Take profit threshold (if applicable)
4. Maximum number of concurrent positions
5. Volatility target or risk per trade values

Consider:
- Strategy volatility and holding period
- Market conditions and asset class
- Risk tolerance and capital constraints
- Correlation between positions

Provide specific numerical values and rationale for each parameter.
"""
        return prompt
    
    @staticmethod
    def get_validation_setup_prompt(
        strategy_description: str,
        data_period_days: int
    ) -> str:
        """
        Generate prompt for cross-validation setup.
        
        Args:
            strategy_description: Description of the strategy
            data_period_days: Total data period in days
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are setting up cross-validation for a quantitative trading strategy.

STRATEGY: {strategy_description}
DATA PERIOD: {data_period_days} days

Recommend cross-validation configuration:
1. Validation type: walk_forward, purged_kfold, or time_series_split
2. Number of splits (considering data size)
3. Training/test split ratios
4. Purge and embargo periods (for purged CV)

Requirements:
- Prevent data leakage between train/test sets
- Ensure sufficient training data in each fold
- Balance between validation robustness and computational cost
- Account for strategy lookback periods

Provide specific parameters and explain the rationale.
"""
        return prompt
    
    @staticmethod
    def get_data_leakage_check_prompt(
        strategy_spec_json: str
    ) -> str:
        """
        Generate prompt for checking data leakage in a strategy.
        
        Args:
            strategy_spec_json: StrategySpec JSON to validate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a data leakage expert reviewing a quantitative trading strategy.

STRATEGY SPEC: {strategy_spec_json}

Check for potential data leakage issues:
1. Do any features use future information?
2. Are lookback periods appropriate for the strategy period?
3. Do entry/exit rules reference future prices?
4. Is the target variable properly lagged?
5. Are there any overlapping time windows?

CRITICAL RULES:
- Features can only use data available at the time of prediction
- Lookback periods must be less than strategy period
- Entry/exit rules must use only past data
- Target must be future returns, not current prices

Identify any issues and suggest fixes to prevent data leakage.
"""
        return prompt
    
    @staticmethod
    def get_strategy_improvement_prompt(
        strategy_spec_json: str,
        performance_metrics: Dict[str, float] = None
    ) -> str:
        """
        Generate prompt for suggesting strategy improvements.
        
        Args:
            strategy_spec_json: Current StrategySpec JSON
            performance_metrics: Performance metrics if available
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a quantitative researcher suggesting improvements for a trading strategy.

CURRENT STRATEGY: {strategy_spec_json}

PERFORMANCE METRICS: {performance_metrics or "Not available"}

Suggest improvements in these areas:
1. Feature Engineering:
   - Additional features for better signal diversity
   - Feature selection to reduce multicollinearity
   - Alternative lookback periods

2. Risk Management:
   - Position sizing optimization
   - Stop loss and take profit adjustments
   - Position limit modifications

3. Entry/Exit Rules:
   - Rule refinement based on market conditions
   - Alternative exit strategies
   - Holding period optimization

4. Validation:
   - Cross-validation improvements
   - Robustness checks
   - Out-of-sample testing

Provide specific, actionable recommendations with rationale.
"""
        return prompt
    
    @staticmethod
    def get_research_note_prompt(
        strategy_spec_json: str,
        backtest_results: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        Generate prompt for creating research notes.
        
        Args:
            strategy_spec_json: StrategySpec JSON
            backtest_results: Backtest results and metrics
            performance_metrics: Performance metrics
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are writing a comprehensive research note for a quantitative trading strategy.

STRATEGY: {strategy_spec_json}
BACKTEST RESULTS: {backtest_results}
PERFORMANCE METRICS: {performance_metrics}

Write a professional research note including:

1. Executive Summary:
   - Strategy overview and key findings
   - Performance highlights and risks

2. Strategy Description:
   - Clear explanation of the investment thesis
   - Feature engineering approach
   - Entry/exit logic

3. Backtest Results:
   - Performance metrics analysis
   - Risk metrics and drawdown analysis
   - Comparison to benchmarks

4. Robustness Analysis:
   - Cross-validation results
   - Parameter sensitivity
   - Market regime analysis

5. Risk Considerations:
   - Data leakage prevention
   - Transaction costs impact
   - Implementation challenges

6. Conclusions and Next Steps:
   - Strategy viability assessment
   - Areas for improvement
   - Future research directions

Use clear, professional language suitable for institutional investors.
"""
        return prompt
    
    @staticmethod
    def get_feature_engineering_prompt(
        base_features: List[str],
        market_context: str = "US equity markets"
    ) -> str:
        """
        Generate prompt for advanced feature engineering.
        
        Args:
            base_features: List of base features
            market_context: Market context for feature engineering
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert in financial feature engineering for {market_context}.

BASE FEATURES: {base_features}

Suggest advanced feature engineering techniques:

1. Technical Indicators:
   - Momentum indicators (RSI, MACD, Stochastic)
   - Volatility indicators (Bollinger Bands, ATR)
   - Trend indicators (Moving averages, ADX)

2. Statistical Features:
   - Rolling statistics (mean, std, skew, kurtosis)
   - Z-scores and percentile ranks
   - Correlation and cointegration measures

3. Market Microstructure:
   - Volume-based features
   - Bid-ask spread proxies
   - Order flow indicators

4. Cross-Asset Features:
   - Sector rotation indicators
   - Currency and commodity relationships
   - Interest rate sensitivity

5. Alternative Data Features:
   - Sentiment indicators
   - Economic calendar events
   - Market regime classifiers

For each feature, specify:
- Name and calculation method
- Appropriate lookback period
- Expected market insight
- Potential data sources

Ensure all features prevent data leakage.
"""
        return prompt
