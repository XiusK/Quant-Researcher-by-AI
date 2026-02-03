"""
Ornstein-Uhlenbeck Process Implementation

Mathematical Form:
    dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

Where:
    kappa: Mean reversion speed (higher = faster reversion)
    theta: Long-term mean level
    sigma: Volatility of the process
    
Recommended for: FX pairs, interest rates, pairs trading
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import optimize
from src.base import BaseStochasticModel, ModelParameters, AssetClass


class OrnsteinUhlenbeck(BaseStochasticModel):
    """
    Mean-reverting stochastic process for stationary time series.
    
    Key Property: E[X_t | X_s] = theta + (X_s - theta) * exp(-kappa * (t-s))
    """
    
    def __init__(self, asset_class: AssetClass = AssetClass.FX):
        """Initialize OU process."""
        super().__init__(asset_class=asset_class, name="Ornstein-Uhlenbeck")
        
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate OU process paths using exact discretization.
        
        Discretization:
            X_{t+dt} = theta + (X_t - theta) * exp(-kappa*dt) + sigma * sqrt((1-exp(-2*kappa*dt))/(2*kappa)) * Z
            where Z ~ N(0,1)
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before simulation")
        
        if seed is not None:
            np.random.seed(seed)
            
        kappa = self.params.params['kappa']
        theta = self.params.params['theta']
        sigma = self.params.params['sigma']
        
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Exact discretization (no Euler bias)
        exp_term = np.exp(-kappa * dt)
        var_term = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
        
        for i in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            paths[:, i + 1] = theta + (paths[:, i] - theta) * exp_term + var_term * Z
            
        return paths
    
    def calibrate(
        self,
        data: pd.Series,
        method: str = "mle"
    ) -> ModelParameters:
        """
        Calibrate OU parameters using Maximum Likelihood Estimation.
        
        Method:
            Maximize log-likelihood assuming transitions are Gaussian
        """
        # Check stationarity first
        from src.base import run_stationarity_tests
        tests = run_stationarity_tests(data)
        
        if not tests['is_stationary']:
            raise ValueError(
                f"Data fails stationarity test (ADF p-value: {tests['adf_pvalue']:.4f}). "
                f"OU process requires stationary data."
            )
        
        # MLE estimation
        X = data.values
        dt = 1.0  # Assume daily data, adjust if needed
        
        def negative_log_likelihood(params: np.ndarray) -> float:
            kappa, theta, sigma = params
            
            if kappa <= 0 or sigma <= 0:
                return 1e10  # Invalid parameters
            
            n = len(X) - 1
            exp_term = np.exp(-kappa * dt)
            var_term = (sigma ** 2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
            
            mean_next = theta + (X[:-1] - theta) * exp_term
            residuals = X[1:] - mean_next
            
            ll = -0.5 * n * np.log(2 * np.pi * var_term) - 0.5 * np.sum(residuals ** 2) / var_term
            return -ll
        
        # Initial guess
        theta_init = np.mean(X)
        kappa_init = 1.0
        sigma_init = np.std(np.diff(X))
        
        result = optimize.minimize(
            negative_log_likelihood,
            x0=[kappa_init, theta_init, sigma_init],
            method='L-BFGS-B',
            bounds=[(0.01, 10), (None, None), (0.001, None)]
        )
        
        if not result.success:
            raise RuntimeError(f"Calibration failed: {result.message}")
        
        kappa_opt, theta_opt, sigma_opt = result.x
        
        self.params = ModelParameters(
            params={
                'kappa': kappa_opt,
                'theta': theta_opt,
                'sigma': sigma_opt
            },
            calibration_window=len(data)
        )
        self.is_calibrated = True
        
        # Add AIC/BIC
        self.params.aic = self.aic(data)
        self.params.bic = self.bic(data)
        
        return self.params
    
    def log_likelihood(self, data: pd.Series) -> float:
        """Calculate log-likelihood of observed data."""
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated first")
        
        X = data.values
        dt = 1.0
        
        kappa = self.params.params['kappa']
        theta = self.params.params['theta']
        sigma = self.params.params['sigma']
        
        n = len(X) - 1
        exp_term = np.exp(-kappa * dt)
        var_term = (sigma ** 2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
        
        mean_next = theta + (X[:-1] - theta) * exp_term
        residuals = X[1:] - mean_next
        
        ll = -0.5 * n * np.log(2 * np.pi * var_term) - 0.5 * np.sum(residuals ** 2) / var_term
        return ll
    
    def half_life(self) -> float:
        """
        Calculate mean reversion half-life.
        
        Returns:
            float: Time periods until half of deviation dissipates
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated first")
        
        return np.log(2) / self.params.params['kappa']
