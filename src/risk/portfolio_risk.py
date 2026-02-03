"""
Portfolio Risk Manager

Enforces hard limits on portfolio exposure and monitors risk metrics in real-time.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.base import BaseRiskManager


class PortfolioRiskManager(BaseRiskManager):
    """
    Real-time portfolio risk monitoring and limit enforcement.
    
    Configuration:
        - max_leverage: Maximum gross leverage (e.g., 2.0)
        - max_position_pct: Maximum single position as % of NAV (e.g., 0.3)
        - max_drawdown: Stop trading if drawdown exceeds (e.g., 0.15)
        - var_limit: Daily VaR limit as % of portfolio (e.g., 0.05)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager with limits."""
        super().__init__(config)
        
        # Set defaults
        self.max_leverage = config.get('max_leverage', 1.0)
        self.max_position_pct = config.get('max_position_pct', 0.3)
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.var_limit = config.get('var_limit', 0.05)
        
        self.peak_nav: float = 0.0
        
    def check_position_limit(
        self,
        proposed_position: float,
        portfolio_value: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify proposed position against risk limits.
        
        Returns:
            (is_allowed, reason_if_blocked)
        """
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive"
        
        # Check leverage
        gross_exposure = abs(proposed_position)
        leverage = gross_exposure / portfolio_value
        
        if leverage > self.max_leverage:
            self.breach_count += 1
            return False, f"Leverage limit breached: {leverage:.2f} > {self.max_leverage}"
        
        # Check position concentration
        position_pct = gross_exposure / portfolio_value
        
        if position_pct > self.max_position_pct:
            self.breach_count += 1
            return False, f"Position size limit breached: {position_pct:.2%} > {self.max_position_pct:.2%}"
        
        return True, None
    
    def check_drawdown(self, current_nav: float) -> Tuple[bool, Optional[str]]:
        """
        Monitor portfolio drawdown and halt trading if limit exceeded.
        
        Returns:
            (can_trade, reason_if_halted)
        """
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav
        
        if self.peak_nav == 0:
            return True, None
        
        drawdown = (self.peak_nav - current_nav) / self.peak_nav
        
        if drawdown > self.max_drawdown:
            return False, f"Drawdown limit breached: {drawdown:.2%} > {self.max_drawdown:.2%}"
        
        return True, None
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk using specified method.
        
        Returns:
            float: VaR as positive number (e.g., 0.05 = 5% potential loss)
        """
        if len(returns) < 50:
            raise ValueError("Insufficient data for VaR calculation (need at least 50 observations)")
        
        if method == "historical":
            # Historical VaR (non-parametric)
            return -np.percentile(returns, (1 - confidence) * 100)
        
        elif method == "parametric":
            # Parametric VaR (assumes normal distribution)
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence)
            return -(returns.mean() + z_score * returns.std())
        
        elif method == "monte_carlo":
            # Monte Carlo VaR (bootstrap)
            n_simulations = 10000
            simulated_returns = np.random.choice(returns, size=n_simulations, replace=True)
            return -np.percentile(simulated_returns, (1 - confidence) * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Returns:
            float: Average loss in worst (1-confidence)% of cases
        """
        var = self.calculate_var(returns, confidence, method="historical")
        threshold = -var
        tail_losses = returns[returns <= threshold]
        
        if len(tail_losses) == 0:
            return var  # Fall back to VaR if no tail observations
        
        return -tail_losses.mean()
    
    def calculate_portfolio_greeks(
        self,
        positions: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks for options exposure.
        
        Note: Simplified implementation for spot positions.
        For options, this should integrate with proper pricing models.
        """
        # For spot positions, only delta is relevant
        total_delta = sum(positions.values())
        
        return {
            'delta': total_delta,
            'gamma': 0.0,  # Zero for spot positions
            'vega': 0.0,   # Zero for spot positions
            'theta': 0.0,  # Zero for spot positions
            'rho': 0.0     # Zero for spot positions
        }
    
    def stress_test(
        self,
        positions: Dict[str, float],
        scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply stress scenarios to current portfolio.
        
        Args:
            positions: Current holdings
            scenarios: Dict of scenario_name -> price_shock_pct
            
        Returns:
            Dict of scenario_name -> portfolio_pnl
        """
        results = {}
        
        for scenario_name, shock in scenarios.items():
            pnl = sum(pos * shock for pos in positions.values())
            results[scenario_name] = pnl
        
        return results
