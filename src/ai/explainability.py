"""
Strategy explainability and research note generation.

This module provides tools for explaining trading strategies and
generating comprehensive research notes using AI.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..strategies.schema import StrategySpec
from ..core.metrics import PerformanceMetrics, RiskMetrics
from .prompts import PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyExplainer:
    """
    Generate explanations and research notes for trading strategies.
    
    This class provides methods to create comprehensive explanations
    of strategies, their performance, and research insights.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the strategy explainer.
        
        Args:
            llm_client: LLM client for generating explanations (optional)
        """
        self.llm_client = llm_client
        self.prompts = PromptTemplates()
        self.performance_metrics = PerformanceMetrics()
        self.risk_metrics = RiskMetrics()
    
    def generate_research_note(
        self,
        strategy_spec: StrategySpec,
        backtest_results: Dict[str, Any],
        performance_metrics: Dict[str, float],
        include_plots: bool = True
    ) -> str:
        """
        Generate a comprehensive research note for a strategy.
        
        Args:
            strategy_spec: Strategy specification
            backtest_results: Backtest results and metrics
            performance_metrics: Performance metrics
            include_plots: Whether to include plot references
            
        Returns:
            Formatted research note as markdown
        """
        logger.info(f"Generating research note for strategy: {strategy_spec.name}")
        
        if self.llm_client:
            # Use LLM to generate research note
            return self._generate_llm_research_note(
                strategy_spec, backtest_results, performance_metrics
            )
        else:
            # Use template-based generation
            return self._generate_template_research_note(
                strategy_spec, backtest_results, performance_metrics, include_plots
            )
    
    def _generate_llm_research_note(
        self,
        strategy_spec: StrategySpec,
        backtest_results: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> str:
        """Generate research note using LLM."""
        try:
            prompt = self.prompts.get_research_note_prompt(
                strategy_spec.to_json(), backtest_results, performance_metrics
            )
            
            response = self.llm_client.generate(prompt)
            return response
            
        except Exception as e:
            logger.error(f"LLM research note generation failed: {e}")
            # Fall back to template generation
            return self._generate_template_research_note(
                strategy_spec, backtest_results, performance_metrics
            )
    
    def _generate_template_research_note(
        self,
        strategy_spec: StrategySpec,
        backtest_results: Dict[str, Any],
        performance_metrics: Dict[str, float],
        include_plots: bool = True
    ) -> str:
        """Generate research note using templates."""
        note = []
        
        # Header
        note.append(f"# {strategy_spec.name}")
        note.append(f"**Research Note** | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        note.append("")
        
        # Executive Summary
        note.append("## Executive Summary")
        note.append(self._generate_executive_summary(strategy_spec, performance_metrics))
        note.append("")
        
        # Strategy Description
        note.append("## Strategy Description")
        note.append(self._generate_strategy_description(strategy_spec))
        note.append("")
        
        # Backtest Results
        note.append("## Backtest Results")
        note.append(self._generate_backtest_results(backtest_results, performance_metrics))
        note.append("")
        
        # Performance Analysis
        note.append("## Performance Analysis")
        note.append(self._generate_performance_analysis(performance_metrics))
        note.append("")
        
        # Risk Analysis
        note.append("## Risk Analysis")
        note.append(self._generate_risk_analysis(performance_metrics))
        note.append("")
        
        # Robustness Analysis
        note.append("## Robustness Analysis")
        note.append(self._generate_robustness_analysis(backtest_results))
        note.append("")
        
        # Implementation Considerations
        note.append("## Implementation Considerations")
        note.append(self._generate_implementation_considerations(strategy_spec))
        note.append("")
        
        # Conclusions
        note.append("## Conclusions and Next Steps")
        note.append(self._generate_conclusions(strategy_spec, performance_metrics))
        note.append("")
        
        # Appendices
        if include_plots:
            note.append("## Appendices")
            note.append(self._generate_plot_references(backtest_results))
            note.append("")
        
        return "\n".join(note)
    
    def _generate_executive_summary(
        self, 
        strategy_spec: StrategySpec, 
        performance_metrics: Dict[str, float]
    ) -> str:
        """Generate executive summary section."""
        summary = []
        
        # Strategy overview
        summary.append(f"This research note presents a quantitative trading strategy called **{strategy_spec.name}**.")
        summary.append(f"The strategy operates on a universe of {len(strategy_spec.universe)} securities over the period {strategy_spec.start_date.strftime('%Y-%m-%d')} to {strategy_spec.end_date.strftime('%Y-%m-%d')}.")
        summary.append("")
        
        # Key performance highlights
        summary.append("### Key Performance Highlights")
        summary.append(f"- **Annualized Return**: {performance_metrics.get('annualized_return', 0):.2%}")
        summary.append(f"- **Sharpe Ratio**: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        summary.append(f"- **Maximum Drawdown**: {performance_metrics.get('max_drawdown', 0):.2%}")
        summary.append(f"- **Win Rate**: {performance_metrics.get('win_rate', 0):.1%}")
        summary.append("")
        
        # Risk assessment
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.0:
            risk_assessment = "The strategy demonstrates strong risk-adjusted returns with a Sharpe ratio above 1.0."
        elif sharpe > 0.5:
            risk_assessment = "The strategy shows moderate risk-adjusted returns."
        else:
            risk_assessment = "The strategy exhibits weak risk-adjusted returns and requires further optimization."
        
        summary.append("### Risk Assessment")
        summary.append(risk_assessment)
        summary.append("")
        
        return "\n".join(summary)
    
    def _generate_strategy_description(self, strategy_spec: StrategySpec) -> str:
        """Generate strategy description section."""
        description = []
        
        # Investment thesis
        description.append("### Investment Thesis")
        description.append(strategy_spec.description)
        description.append("")
        
        # Feature engineering
        description.append("### Feature Engineering")
        description.append(f"The strategy utilizes {len(strategy_spec.features)} features:")
        for feature in strategy_spec.features:
            description.append(f"- **{feature.name}**: {feature.feature_type.value} feature with {feature.lookback_period}-day lookback")
        description.append("")
        
        # Entry/exit logic
        description.append("### Entry/Exit Logic")
        description.append(f"**Entry Rules**: {strategy_spec.entry_rules.condition}")
        description.append(f"**Exit Rules**: {strategy_spec.exit_rules.condition}")
        description.append(f"**Holding Period**: Maximum {strategy_spec.holding_period} days")
        description.append("")
        
        # Risk management
        description.append("### Risk Management")
        description.append(f"**Position Sizing**: {strategy_spec.position_sizing.method}")
        if strategy_spec.position_sizing.volatility_target:
            description.append(f"**Volatility Target**: {strategy_spec.position_sizing.volatility_target:.1%}")
        if strategy_spec.position_sizing.risk_per_trade:
            description.append(f"**Risk Per Trade**: {strategy_spec.position_sizing.risk_per_trade:.1%}")
        description.append(f"**Max Positions**: {strategy_spec.max_positions}")
        description.append("")
        
        return "\n".join(description)
    
    def _generate_backtest_results(
        self, 
        backtest_results: Dict[str, Any], 
        performance_metrics: Dict[str, float]
    ) -> str:
        """Generate backtest results section."""
        results = []
        
        # Performance overview
        results.append("### Performance Overview")
        results.append(f"- **Total Return**: {performance_metrics.get('total_return', 0):.2%}")
        results.append(f"- **Annualized Return**: {performance_metrics.get('annualized_return', 0):.2%}")
        results.append(f"- **Volatility**: {performance_metrics.get('volatility', 0):.2%}")
        results.append("")
        
        # Risk-adjusted metrics
        results.append("### Risk-Adjusted Metrics")
        results.append(f"- **Sharpe Ratio**: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        results.append(f"- **Sortino Ratio**: {performance_metrics.get('sortino_ratio', 0):.2f}")
        results.append(f"- **Calmar Ratio**: {performance_metrics.get('calmar_ratio', 0):.2f}")
        results.append("")
        
        # Trading statistics
        results.append("### Trading Statistics")
        results.append(f"- **Win Rate**: {performance_metrics.get('win_rate', 0):.1%}")
        results.append(f"- **Profit Factor**: {performance_metrics.get('profit_factor', 0):.2f}")
        results.append(f"- **Hit Rate**: {performance_metrics.get('hit_rate', 0):.1%}")
        if 'turnover' in performance_metrics:
            results.append(f"- **Annual Turnover**: {performance_metrics['turnover']:.1%}")
        results.append("")
        
        return "\n".join(results)
    
    def _generate_performance_analysis(self, performance_metrics: Dict[str, float]) -> str:
        """Generate performance analysis section."""
        analysis = []
        
        # Return analysis
        analysis.append("### Return Analysis")
        annual_return = performance_metrics.get('annualized_return', 0)
        if annual_return > 0.15:
            analysis.append("The strategy generates strong annualized returns above 15%, indicating effective alpha generation.")
        elif annual_return > 0.08:
            analysis.append("The strategy produces solid annualized returns above 8%, showing consistent performance.")
        else:
            analysis.append("The strategy's annualized returns are below 8%, suggesting room for improvement.")
        analysis.append("")
        
        # Risk-adjusted performance
        analysis.append("### Risk-Adjusted Performance")
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            analysis.append("Exceptional risk-adjusted performance with Sharpe ratio above 1.5.")
        elif sharpe > 1.0:
            analysis.append("Strong risk-adjusted performance with Sharpe ratio above 1.0.")
        elif sharpe > 0.5:
            analysis.append("Moderate risk-adjusted performance with room for optimization.")
        else:
            analysis.append("Weak risk-adjusted performance requiring significant improvement.")
        analysis.append("")
        
        # Distribution analysis
        analysis.append("### Return Distribution")
        skewness = performance_metrics.get('skewness', 0)
        kurtosis = performance_metrics.get('kurtosis', 0)
        
        if skewness > 0.5:
            analysis.append("Returns show positive skewness, indicating more frequent large positive returns.")
        elif skewness < -0.5:
            analysis.append("Returns show negative skewness, indicating more frequent large negative returns.")
        else:
            analysis.append("Returns show relatively symmetric distribution.")
        
        if kurtosis > 5:
            analysis.append("High kurtosis indicates frequent extreme returns (both positive and negative).")
        else:
            analysis.append("Moderate kurtosis suggests relatively stable return distribution.")
        analysis.append("")
        
        return "\n".join(analysis)
    
    def _generate_risk_analysis(self, performance_metrics: Dict[str, float]) -> str:
        """Generate risk analysis section."""
        analysis = []
        
        # Drawdown analysis
        analysis.append("### Drawdown Analysis")
        max_dd = performance_metrics.get('max_drawdown', 0)
        if max_dd < -0.1:
            analysis.append(f"**Maximum Drawdown**: {max_dd:.1%} - The strategy experienced significant drawdowns.")
        elif max_dd < -0.05:
            analysis.append(f"**Maximum Drawdown**: {max_dd:.1%} - Moderate drawdowns within acceptable limits.")
        else:
            analysis.append(f"**Maximum Drawdown**: {max_dd:.1%} - Low drawdowns indicating good risk control.")
        analysis.append("")
        
        # Value at Risk
        analysis.append("### Value at Risk")
        var_95 = performance_metrics.get('var_95', 0)
        cvar_95 = performance_metrics.get('cvar_95', 0)
        analysis.append(f"**95% VaR**: {var_95:.2%} - 95% of daily returns are above this threshold.")
        analysis.append(f"**95% CVaR**: {cvar_95:.2%} - Expected loss on worst 5% of days.")
        analysis.append("")
        
        # Volatility analysis
        analysis.append("### Volatility Analysis")
        volatility = performance_metrics.get('volatility', 0)
        if volatility > 0.25:
            analysis.append("High volatility suggests aggressive strategy with significant price swings.")
        elif volatility > 0.15:
            analysis.append("Moderate volatility indicates balanced risk-return profile.")
        else:
            analysis.append("Low volatility suggests conservative strategy with stable returns.")
        analysis.append("")
        
        return "\n".join(analysis)
    
    def _generate_robustness_analysis(self, backtest_results: Dict[str, Any]) -> str:
        """Generate robustness analysis section."""
        analysis = []
        
        # Cross-validation results
        if 'cv_results' in backtest_results:
            cv_results = backtest_results['cv_results']
            analysis.append("### Cross-Validation Results")
            analysis.append(f"**Validation Method**: {cv_results.get('cv_method', 'Unknown')}")
            analysis.append(f"**Number of Folds**: {cv_results.get('n_splits', 0)}")
            analysis.append(f"**Mean Score**: {cv_results.get('mean_score', 0):.3f}")
            analysis.append(f"**Score Std**: {cv_results.get('std_score', 0):.3f}")
            analysis.append("")
        
        # Out-of-sample performance
        if 'oos_results' in backtest_results:
            oos_results = backtest_results['oos_results']
            analysis.append("### Out-of-Sample Performance")
            analysis.append(f"**OOS Period**: {oos_results.get('start_date', 'Unknown')} to {oos_results.get('end_date', 'Unknown')}")
            analysis.append(f"**OOS Return**: {oos_results.get('return', 0):.2%}")
            analysis.append(f"**OOS Sharpe**: {oos_results.get('sharpe', 0):.2f}")
            analysis.append("")
        
        # Parameter sensitivity
        analysis.append("### Parameter Sensitivity")
        analysis.append("The strategy's performance should be tested across different parameter combinations:")
        analysis.append("- Feature lookback periods")
        analysis.append("- Entry/exit thresholds")
        analysis.append("- Position sizing parameters")
        analysis.append("- Holding period variations")
        analysis.append("")
        
        return "\n".join(analysis)
    
    def _generate_implementation_considerations(self, strategy_spec: StrategySpec) -> str:
        """Generate implementation considerations section."""
        considerations = []
        
        # Data requirements
        considerations.append("### Data Requirements")
        max_lookback = strategy_spec.get_max_lookback()
        considerations.append(f"**Minimum Data**: {max_lookback + 252} days of historical data required")
        considerations.append(f"**Features**: {len(strategy_spec.features)} features with {max_lookback}-day maximum lookback")
        considerations.append("")
        
        # Transaction costs
        considerations.append("### Transaction Costs")
        considerations.append("**Implementation Impact**:")
        considerations.append("- Bid-ask spreads on entry/exit")
        considerations.append("- Commission costs per trade")
        considerations.append("- Market impact for large positions")
        considerations.append("- Slippage during volatile periods")
        considerations.append("")
        
        # Operational considerations
        considerations.append("### Operational Considerations")
        considerations.append("**Risk Management**:")
        considerations.append("- Real-time position monitoring")
        considerations.append("- Stop-loss enforcement")
        considerations.append("- Position size limits")
        considerations.append("- Correlation monitoring")
        considerations.append("")
        
        return "\n".join(considerations)
    
    def _generate_conclusions(
        self, 
        strategy_spec: StrategySpec, 
        performance_metrics: Dict[str, float]
    ) -> str:
        """Generate conclusions section."""
        conclusions = []
        
        # Strategy assessment
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        max_dd = performance_metrics.get('max_drawdown', 0)
        
        conclusions.append("### Strategy Assessment")
        if sharpe > 1.0 and max_dd > -0.15:
            conclusions.append("**Viable Strategy**: The strategy demonstrates strong risk-adjusted returns with acceptable drawdowns.")
        elif sharpe > 0.5 and max_dd > -0.25:
            conclusions.append("**Promising Strategy**: The strategy shows potential but requires optimization for production use.")
        else:
            conclusions.append("**Development Required**: The strategy needs significant improvement before consideration.")
        conclusions.append("")
        
        # Areas for improvement
        conclusions.append("### Areas for Improvement")
        if performance_metrics.get('win_rate', 0) < 0.5:
            conclusions.append("- Improve entry/exit signal quality")
        if performance_metrics.get('sharpe_ratio', 0) < 1.0:
            conclusions.append("- Optimize risk-adjusted returns")
        if performance_metrics.get('max_drawdown', 0) < -0.2:
            conclusions.append("- Enhance risk management controls")
        conclusions.append("- Conduct additional robustness testing")
        conclusions.append("")
        
        # Next steps
        conclusions.append("### Next Steps")
        conclusions.append("1. **Parameter Optimization**: Use grid search or Bayesian optimization")
        conclusions.append("2. **Feature Engineering**: Explore additional predictive features")
        conclusions.append("3. **Risk Management**: Implement dynamic position sizing")
        conclusions.append("4. **Market Regime Analysis**: Test performance across different market conditions")
        conclusions.append("5. **Live Testing**: Consider paper trading before live implementation")
        conclusions.append("")
        
        return "\n".join(conclusions)
    
    def _generate_plot_references(self, backtest_results: Dict[str, Any]) -> str:
        """Generate plot references section."""
        plots = []
        
        plots.append("### Performance Charts")
        plots.append("The following charts provide visual analysis of the strategy:")
        plots.append("")
        
        if 'equity_curve' in backtest_results:
            plots.append("- **Equity Curve**: Portfolio value over time")
        if 'drawdown' in backtest_results:
            plots.append("- **Drawdown Chart**: Peak-to-trough declines")
        if 'returns_distribution' in backtest_results:
            plots.append("- **Returns Distribution**: Histogram of daily returns")
        if 'rolling_metrics' in backtest_results:
            plots.append("- **Rolling Metrics**: Time-varying performance measures")
        
        plots.append("")
        plots.append("### Feature Analysis")
        plots.append("- **Feature Importance**: SHAP values for model interpretability")
        plots.append("- **Feature Correlation**: Multicollinearity analysis")
        plots.append("- **Feature Stability**: Time-varying feature behavior")
        
        return "\n".join(plots)
    
    def explain_strategy_decision(
        self, 
        strategy_spec: StrategySpec,
        decision_date: datetime,
        market_data: Dict[str, Any]
    ) -> str:
        """
        Explain a specific strategy decision on a given date.
        
        Args:
            strategy_spec: Strategy specification
            decision_date: Date of the decision
            market_data: Market data for the decision date
            
        Returns:
            Explanation of the decision
        """
        explanation = []
        
        explanation.append(f"## Strategy Decision Explanation - {decision_date.strftime('%Y-%m-%d')}")
        explanation.append("")
        
        # Feature values
        explanation.append("### Feature Values")
        for feature in strategy_spec.features:
            if feature.name in market_data:
                value = market_data[feature.name]
                explanation.append(f"- **{feature.name}**: {value:.4f}")
        explanation.append("")
        
        # Decision logic
        explanation.append("### Decision Logic")
        explanation.append(f"**Entry Condition**: {strategy_spec.entry_rules.condition}")
        explanation.append(f"**Exit Condition**: {strategy_spec.exit_rules.condition}")
        explanation.append("")
        
        # Market context
        explanation.append("### Market Context")
        if 'market_regime' in market_data:
            explanation.append(f"**Market Regime**: {market_data['market_regime']}")
        if 'volatility' in market_data:
            explanation.append(f"**Market Volatility**: {market_data['volatility']:.2%}")
        explanation.append("")
        
        return "\n".join(explanation)
    
    def generate_strategy_comparison(
        self, 
        strategies: List[StrategySpec],
        performance_metrics: List[Dict[str, float]]
    ) -> str:
        """
        Generate comparison of multiple strategies.
        
        Args:
            strategies: List of strategy specifications
            performance_metrics: List of performance metrics for each strategy
            
        Returns:
            Strategy comparison report
        """
        comparison = []
        
        comparison.append("# Strategy Comparison Report")
        comparison.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comparison.append("")
        
        # Summary table
        comparison.append("## Performance Summary")
        comparison.append("| Strategy | Return | Sharpe | Max DD | Win Rate |")
        comparison.append("|----------|---------|---------|---------|----------|")
        
        for i, (strategy, metrics) in enumerate(zip(strategies, performance_metrics)):
            row = f"| {strategy.name} | "
            row += f"{metrics.get('annualized_return', 0):.2%} | "
            row += f"{metrics.get('sharpe_ratio', 0):.2f} | "
            row += f"{metrics.get('max_drawdown', 0):.2%} | "
            row += f"{metrics.get('win_rate', 0):.1%} |"
            comparison.append(row)
        
        comparison.append("")
        
        # Detailed analysis
        comparison.append("## Detailed Analysis")
        for i, (strategy, metrics) in enumerate(zip(strategies, performance_metrics)):
            comparison.append(f"### {strategy.name}")
            comparison.append(f"**Description**: {strategy.description}")
            comparison.append(f"**Universe**: {len(strategy.universe)} securities")
            comparison.append(f"**Features**: {len(strategy.features)} features")
            comparison.append(f"**Holding Period**: {strategy.holding_period} days")
            comparison.append("")
            
            # Performance highlights
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe == max(m.get('sharpe_ratio', 0) for m in performance_metrics):
                comparison.append("**Best Sharpe Ratio** among compared strategies")
            comparison.append("")
        
        return "\n".join(comparison)
