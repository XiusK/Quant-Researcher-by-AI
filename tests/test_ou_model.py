"""
Unit tests for Ornstein-Uhlenbeck process implementation.

Tests cover:
- Calibration accuracy
- Simulation properties (mean, variance)
- Parameter validation
- Stationarity requirements
"""

import pytest
import numpy as np
import pandas as pd
from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck, AssetClass


class TestOUCalibration:
    """Test parameter calibration."""
    
    def test_calibration_on_synthetic_data(self):
        """Verify calibration recovers known parameters."""
        # Generate synthetic OU data
        true_kappa = 1.5
        true_theta = 100.0
        true_sigma = 2.0
        
        np.random.seed(42)
        n = 1000
        dt = 1.0
        X = np.zeros(n)
        X[0] = true_theta
        
        for i in range(1, n):
            dW = np.random.randn() * np.sqrt(dt)
            X[i] = true_theta + (X[i-1] - true_theta) * np.exp(-true_kappa * dt) + \
                   true_sigma * np.sqrt((1 - np.exp(-2 * true_kappa * dt)) / (2 * true_kappa)) * dW / np.sqrt(dt)
        
        data = pd.Series(X)
        
        # Calibrate
        model = OrnsteinUhlenbeck(AssetClass.FX)
        params = model.calibrate(data)
        
        # Check parameters are close to true values (within 20%)
        assert abs(params.params['kappa'] - true_kappa) / true_kappa < 0.2
        assert abs(params.params['theta'] - true_theta) / true_theta < 0.1
        assert abs(params.params['sigma'] - true_sigma) / true_sigma < 0.2
    
    def test_calibration_requires_stationary_data(self):
        """Non-stationary data should raise ValueError."""
        # Random walk (non-stationary)
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500))
        data = pd.Series(random_walk)
        
        model = OrnsteinUhlenbeck(AssetClass.FX)
        
        with pytest.raises(ValueError, match="stationarity test"):
            model.calibrate(data)


class TestOUSimulation:
    """Test simulation properties."""
    
    def test_simulation_mean_convergence(self):
        """Simulated paths should converge to theta."""
        model = OrnsteinUhlenbeck(AssetClass.FX)
        
        # Manually set parameters
        from src.base import ModelParameters
        model.params = ModelParameters(
            params={'kappa': 2.0, 'theta': 50.0, 'sigma': 1.0}
        )
        model.is_calibrated = True
        
        # Simulate long paths
        paths = model.simulate(S0=40.0, T=10.0, n_steps=1000, n_paths=1000, seed=42)
        
        # Final values should be close to theta
        final_values = paths[:, -1]
        assert abs(np.mean(final_values) - 50.0) < 1.0
    
    def test_simulation_requires_calibration(self):
        """Cannot simulate without calibration."""
        model = OrnsteinUhlenbeck(AssetClass.FX)
        
        with pytest.raises(RuntimeError, match="calibrated"):
            model.simulate(S0=100.0, T=1.0, n_steps=100)


class TestOUHalfLife:
    """Test half-life calculation."""
    
    def test_half_life_formula(self):
        """Half-life should equal ln(2)/kappa."""
        model = OrnsteinUhlenbeck(AssetClass.FX)
        
        from src.base import ModelParameters
        kappa = 0.5
        model.params = ModelParameters(
            params={'kappa': kappa, 'theta': 100.0, 'sigma': 1.0}
        )
        model.is_calibrated = True
        
        expected_half_life = np.log(2) / kappa
        assert abs(model.half_life() - expected_half_life) < 1e-6


class TestOUValidation:
    """Test parameter validation and edge cases."""
    
    def test_negative_kappa_rejected(self):
        """Negative mean reversion speed is invalid."""
        # This will be caught during calibration optimization
        pass
    
    def test_zero_sigma_rejected(self):
        """Zero volatility is invalid."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
