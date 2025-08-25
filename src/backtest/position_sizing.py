"""
Position sizing strategies for backtesting.

This module provides various position sizing methods to manage
risk and optimize portfolio allocation in trading strategies.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Base position sizing class for backtesting.
    
    This class provides various methods to calculate position sizes
    based on different risk management approaches.
    """
    
    def __init__(self, max_position_size: float = 0.1):
        """
        Initialize position sizer.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio (default: 10%)
        """
        self.max_position_size = max_position_size
        logger.info(f"Position sizer initialized with max position size: {max_position_size:.1%}")
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        method: str = "fixed_fraction",
        **kwargs
    ) -> int:
        """
        Calculate position size using specified method.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            method: Position sizing method
            **kwargs: Additional parameters for the method
            
        Returns:
            Number of shares to trade
        """
        methods = {
            "fixed_fraction": self._fixed_fraction_sizing,
            "volatility_target": self._volatility_target_sizing,
            "risk_per_trade": self._risk_per_trade_sizing,
            "kelly_criterion": self._kelly_criterion_sizing,
            "equal_weight": self._equal_weight_sizing,
            "market_cap_weight": self._market_cap_weight_sizing
        }
        
        if method not in methods:
            logger.warning(f"Unknown position sizing method: {method}. Using fixed_fraction.")
            method = "fixed_fraction"
        
        return methods[method](portfolio_value, price, **kwargs)
    
    def _fixed_fraction_sizing(
        self,
        portfolio_value: float,
        price: float,
        fraction: Optional[float] = None
    ) -> int:
        """
        Fixed fraction position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            fraction: Fraction of portfolio to allocate (defaults to max_position_size)
            
        Returns:
            Number of shares
        """
        fraction = fraction or self.max_position_size
        target_value = portfolio_value * fraction
        shares = int(target_value / price)
        
        logger.debug(f"Fixed fraction sizing: {fraction:.1%} of ${portfolio_value:,.2f} = {shares} shares")
        return shares
    
    def _volatility_target_sizing(
        self,
        portfolio_value: float,
        price: float,
        volatility_target: float = 0.15,
        asset_volatility: float = 0.20,
        risk_free_rate: float = 0.02
    ) -> int:
        """
        Volatility target position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            volatility_target: Target portfolio volatility (annualized)
            asset_volatility: Asset volatility (annualized)
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Number of shares
        """
        # Calculate optimal weight based on volatility target
        # This is a simplified version - in practice, you'd use more sophisticated methods
        optimal_weight = volatility_target / asset_volatility
        
        # Apply constraints
        optimal_weight = min(optimal_weight, self.max_position_size)
        optimal_weight = max(optimal_weight, 0.01)  # Minimum 1%
        
        target_value = portfolio_value * optimal_weight
        shares = int(target_value / price)
        
        logger.debug(f"Volatility target sizing: {optimal_weight:.1%} weight = {shares} shares")
        return shares
    
    def _risk_per_trade_sizing(
        self,
        portfolio_value: float,
        price: float,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05
    ) -> int:
        """
        Risk per trade position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            risk_per_trade: Maximum risk per trade as fraction of portfolio
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Number of shares
        """
        # Calculate maximum dollar risk per trade
        max_dollar_risk = portfolio_value * risk_per_trade
        
        # Calculate position size based on stop loss
        risk_per_share = price * stop_loss_pct
        shares = int(max_dollar_risk / risk_per_share)
        
        # Apply maximum position size constraint
        max_shares = int((portfolio_value * self.max_position_size) / price)
        shares = min(shares, max_shares)
        
        logger.debug(f"Risk per trade sizing: {shares} shares (risk: ${max_dollar_risk:,.2f})")
        return shares
    
    def _kelly_criterion_sizing(
        self,
        portfolio_value: float,
        price: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015
    ) -> int:
        """
        Kelly criterion position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            
        Returns:
            Number of shares
        """
        # Calculate Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply constraints (typically use half Kelly for safety)
        kelly_fraction = kelly_fraction * 0.5
        kelly_fraction = min(kelly_fraction, self.max_position_size)
        kelly_fraction = max(kelly_fraction, 0.01)  # Minimum 1%
        
        target_value = portfolio_value * kelly_fraction
        shares = int(target_value / price)
        
        logger.debug(f"Kelly criterion sizing: {kelly_fraction:.1%} weight = {shares} shares")
        return shares
    
    def _equal_weight_sizing(
        self,
        portfolio_value: float,
        price: float,
        num_positions: int = 10
    ) -> int:
        """
        Equal weight position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            num_positions: Target number of positions
            
        Returns:
            Number of shares
        """
        # Equal weight allocation
        weight_per_position = 1.0 / num_positions
        
        # Apply maximum position size constraint
        weight_per_position = min(weight_per_position, self.max_position_size)
        
        target_value = portfolio_value * weight_per_position
        shares = int(target_value / price)
        
        logger.debug(f"Equal weight sizing: {weight_per_position:.1%} weight = {shares} shares")
        return shares
    
    def _market_cap_weight_sizing(
        self,
        portfolio_value: float,
        price: float,
        market_cap: float,
        total_market_cap: float,
        min_weight: float = 0.01,
        max_weight: Optional[float] = None
    ) -> int:
        """
        Market cap weighted position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            market_cap: Asset market capitalization
            total_market_cap: Total market capitalization of universe
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint (defaults to max_position_size)
            
        Returns:
            Number of shares
        """
        max_weight = max_weight or self.max_position_size
        
        # Calculate market cap weight
        market_cap_weight = market_cap / total_market_cap
        
        # Apply constraints
        market_cap_weight = max(market_cap_weight, min_weight)
        market_cap_weight = min(market_cap_weight, max_weight)
        
        target_value = portfolio_value * market_cap_weight
        shares = int(target_value / price)
        
        logger.debug(f"Market cap weight sizing: {market_cap_weight:.1%} weight = {shares} shares")
        return shares


class RiskAdjustedPositionSizer(PositionSizer):
    """
    Risk-adjusted position sizing with advanced risk management.
    
    This class extends the base position sizer with additional
    risk management features.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_portfolio_risk: float = 0.02,
        correlation_threshold: float = 0.7
    ):
        """
        Initialize risk-adjusted position sizer.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk (VaR)
            correlation_threshold: Maximum correlation threshold for positions
        """
        super().__init__(max_position_size)
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_threshold = correlation_threshold
        
        # Track current positions for risk management
        self.current_positions = {}
        self.position_correlations = {}
    
    def calculate_risk_adjusted_position_size(
        self,
        portfolio_value: float,
        price: float,
        asset_volatility: float,
        correlations: Optional[Dict[str, float]] = None,
        method: str = "risk_parity",
        **kwargs
    ) -> int:
        """
        Calculate risk-adjusted position size.
        
        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            asset_volatility: Asset volatility
            correlations: Correlation with existing positions
            method: Risk adjustment method
            **kwargs: Additional parameters
            
        Returns:
            Risk-adjusted number of shares
        """
        # Calculate base position size
        base_shares = self.calculate_position_size(portfolio_value, price, method, **kwargs)
        
        # Apply risk adjustments
        adjusted_shares = self._apply_risk_adjustments(
            base_shares, asset_volatility, correlations, portfolio_value
        )
        
        return adjusted_shares
    
    def _apply_risk_adjustments(
        self,
        base_shares: int,
        asset_volatility: float,
        correlations: Optional[Dict[str, float]] = None,
        portfolio_value: float = 0.0
    ) -> int:
        """
        Apply risk adjustments to position size.
        
        Args:
            base_shares: Base position size
            asset_volatility: Asset volatility
            correlations: Correlation with existing positions
            portfolio_value: Portfolio value for risk calculations
            
        Returns:
            Risk-adjusted position size
        """
        adjusted_shares = base_shares
        
        # Correlation adjustment
        if correlations:
            max_correlation = max(correlations.values()) if correlations.values() else 0
            if max_correlation > self.correlation_threshold:
                # Reduce position size for highly correlated assets
                correlation_factor = 1 - (max_correlation - self.correlation_threshold)
                correlation_factor = max(correlation_factor, 0.5)  # Minimum 50% reduction
                adjusted_shares = int(adjusted_shares * correlation_factor)
                
                logger.debug(f"Correlation adjustment: {correlation_factor:.2f} factor")
        
        # Portfolio risk adjustment
        if portfolio_value > 0:
            current_risk = self._calculate_portfolio_risk()
            new_position_risk = (adjusted_shares * asset_volatility) / portfolio_value
            
            if current_risk + new_position_risk > self.max_portfolio_risk:
                # Reduce position to stay within risk limits
                risk_reduction = (self.max_portfolio_risk - current_risk) / new_position_risk
                risk_reduction = max(risk_reduction, 0.1)  # Minimum 10% of original
                adjusted_shares = int(adjusted_shares * risk_reduction)
                
                logger.debug(f"Risk adjustment: {risk_reduction:.2f} factor")
        
        return adjusted_shares
    
    def _calculate_portfolio_risk(self) -> float:
        """
        Calculate current portfolio risk.
        
        Returns:
            Current portfolio risk measure
        """
        # Simplified portfolio risk calculation
        # In practice, you'd use more sophisticated methods like VaR or CVaR
        total_risk = 0.0
        
        for position in self.current_positions.values():
            total_risk += position.get('risk_contribution', 0.0)
        
        return total_risk
    
    def add_position(self, symbol: str, shares: int, volatility: float, **kwargs):
        """
        Add a position to the tracker.
        
        Args:
            symbol: Asset symbol
            shares: Number of shares
            volatility: Asset volatility
            **kwargs: Additional position information
        """
        self.current_positions[symbol] = {
            'shares': shares,
            'volatility': volatility,
            'risk_contribution': shares * volatility,
            **kwargs
        }
    
    def remove_position(self, symbol: str):
        """
        Remove a position from the tracker.
        
        Args:
            symbol: Asset symbol
        """
        if symbol in self.current_positions:
            del self.current_positions[symbol]
    
    def update_correlations(self, symbol: str, correlations: Dict[str, float]):
        """
        Update position correlations.
        
        Args:
            symbol: Asset symbol
            correlations: Correlation with other assets
        """
        self.position_correlations[symbol] = correlations


def create_position_sizer(
    sizer_type: str = "basic",
    max_position_size: float = 0.1,
    **kwargs
) -> PositionSizer:
    """
    Factory function to create position sizers.
    
    Args:
        sizer_type: Type of position sizer
        max_position_size: Maximum position size
        **kwargs: Additional parameters
        
    Returns:
        Configured position sizer instance
    """
    sizers = {
        "basic": PositionSizer,
        "risk_adjusted": RiskAdjustedPositionSizer
    }
    
    if sizer_type not in sizers:
        logger.warning(f"Unknown position sizer type: {sizer_type}. Using basic sizer.")
        sizer_type = "basic"
    
    sizer_class = sizers[sizer_type]
    return sizer_class(max_position_size=max_position_size, **kwargs)
