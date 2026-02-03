"""
Jump-Diffusion Model for Gold

Mathematical Form:
    dS_t = μS_t dt + σS_t dW_t + S_t dJ_t
    
Where:
    - μ: Drift coefficient
    - σ: Diffusion volatility
    - J_t: Compound Poisson jump process with intensity λ
    - Jump size ~ N(μ_J, σ_J²)

Rationale:
    Gold exhibits frequent jumps during:
    - Fed announcements
    - Geopolitical events
    - Crisis periods
    - Market crashes (safe haven flows)

References:
    Merton, R. C. (1976). "Option pricing when underlying stock returns are discontinuous"
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from src.base import BaseStochasticModel, ModelParameters, AssetClass


class JumpDiffusionModel(BaseStochasticModel):
    """
    Merton Jump-Diffusion model for assets with discontinuous jumps.
    
    Suitable for: Crypto, Gold, Event-driven strategies
    """
    
    def __init__(self, asset_class: AssetClass = AssetClass.COMMODITIES):
        """Initialize Jump-Diffusion model."""
        super().__init__(asset_class=asset_class, name="Jump-Diffusion")
        
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate jump-diffusion paths using Euler-Maruyama scheme.
        
        Discretization:
            S_{t+dt} = S_t * exp((μ - 0.5σ²)dt + σ√dt*Z + J*N_t)
            
        Where:
            Z ~ N(0,1) (Brownian motion)
            N_t ~ Poisson(λ*dt) (jump count)
            J ~ N(μ_J, σ_J²) (jump size)
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        mu = self.params.params['mu']
        sigma = self.params.params['sigma']
        lambda_jump = self.params.params['lambda_jump']
        mu_jump = self.params.params['mu_jump']
        sigma_jump = self.params.params['sigma_jump']
        
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_steps):
            # Diffusion component
            Z = np.random.standard_normal(n_paths)
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            
            # Jump component
            n_jumps = np.random.poisson(lambda_jump * dt, n_paths)
            jump_component = np.zeros(n_paths)
            
            for j in range(n_paths):
                if n_jumps[j] > 0:
                    jumps = np.random.normal(mu_jump, sigma_jump, n_jumps[j])
                    jump_component[j] = np.sum(jumps)
            
            # Update price
            paths[:, i + 1] = paths[:, i] * np.exp(diffusion + jump_component)
        
        return paths
    
    def calibrate(
        self,
        data: pd.Series,
        method: str = "mle"
    ) -> ModelParameters:
        """
        Calibrate jump-diffusion parameters using Maximum Likelihood.
        
        Method:
            1. Identify jumps (returns > 3σ threshold)
            2. Separate diffusion and jump components
            3. Estimate parameters via MLE
        """
        returns = np.log(data / data.shift(1)).dropna()
        
        # Initial detection of jumps (simple threshold method)
        std_returns = returns.std()
        jump_threshold = 3 * std_returns
        
        is_jump = np.abs(returns) > jump_threshold
        n_jumps = is_jump.sum()
        
        if n_jumps < 5:
            raise ValueError(
                f"Insufficient jumps detected ({n_jumps}). "
                f"Jump-Diffusion may not be appropriate for this data."
            )
        
        print(f"Detected {n_jumps} jumps out of {len(returns)} observations ({n_jumps/len(returns):.2%})")
        
        # Separate jump and diffusion returns
        diffusion_returns = returns[~is_jump]
        jump_returns = returns[is_jump]
        
        # Estimate diffusion parameters from non-jump days
        mu_diffusion = diffusion_returns.mean() * 252  # Annualized
        sigma_diffusion = diffusion_returns.std() * np.sqrt(252)
        
        # Estimate jump parameters
        lambda_jump = n_jumps / len(returns) * 252  # Jumps per year
        mu_jump = jump_returns.mean()
        sigma_jump = jump_returns.std()
        
        # MLE refinement (simplified)
        def negative_log_likelihood(params: np.ndarray) -> float:
            mu, sigma, lam, mu_j, sigma_j = params
            
            if sigma <= 0 or sigma_j <= 0 or lam <= 0:
                return 1e10
            
            # Log-likelihood for diffusion component
            ll_diffusion = -0.5 * np.sum(
                ((diffusion_returns - mu/252) / (sigma/np.sqrt(252))) ** 2
            ) - len(diffusion_returns) * np.log(sigma)
            
            # Log-likelihood for jump component (simplified)
            ll_jumps = -0.5 * np.sum(
                ((jump_returns - mu_j) / sigma_j) ** 2
            ) - n_jumps * np.log(sigma_j)
            
            # Poisson likelihood for jump timing
            ll_poisson = n_jumps * np.log(lam / 252) - (lam / 252) * len(returns)
            
            return -(ll_diffusion + ll_jumps + ll_poisson)
        
        # Optimize
        initial_guess = [mu_diffusion, sigma_diffusion, lambda_jump, mu_jump, sigma_jump]
        
        result = optimize.minimize(
            negative_log_likelihood,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=[
                (-0.5, 0.5),      # mu
                (0.01, 2.0),      # sigma
                (0.1, 100),       # lambda
                (-0.2, 0.2),      # mu_jump
                (0.001, 0.5)      # sigma_jump
            ]
        )
        
        if not result.success:
            print(f"Warning: Optimization did not fully converge: {result.message}")
        
        mu_opt, sigma_opt, lambda_opt, mu_j_opt, sigma_j_opt = result.x
        
        self.params = ModelParameters(
            params={
                'mu': mu_opt,
                'sigma': sigma_opt,
                'lambda_jump': lambda_opt,
                'mu_jump': mu_j_opt,
                'sigma_jump': sigma_j_opt
            },
            calibration_window=len(data)
        )
        self.is_calibrated = True
        
        # Add AIC/BIC
        self.params.aic = self.aic(data)
        self.params.bic = self.bic(data)
        
        return self.params
    
    def log_likelihood(self, data: pd.Series) -> float:
        """Calculate log-likelihood (simplified approximation)."""
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated first")
        
        returns = np.log(data / data.shift(1)).dropna()
        
        mu = self.params.params['mu']
        sigma = self.params.params['sigma']
        
        # Simplified: Use normal likelihood (underestimate)
        ll = -0.5 * len(returns) * np.log(2 * np.pi * (sigma/np.sqrt(252))**2)
        ll -= 0.5 * np.sum(((returns - mu/252) / (sigma/np.sqrt(252))) ** 2)
        
        return ll
    
    def expected_jump_size(self) -> float:
        """Calculate expected jump magnitude."""
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated first")
        
        mu_j = self.params.params['mu_jump']
        sigma_j = self.params.params['sigma_jump']
        
        # Expected absolute jump size
        return np.abs(mu_j) + sigma_j
    
    def jump_frequency_per_year(self) -> float:
        """Return expected number of jumps per year."""
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated first")
        
        return self.params.params['lambda_jump']
