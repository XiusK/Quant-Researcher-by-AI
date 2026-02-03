"""
Hidden Markov Model for Market Regime Detection

Mathematical Foundation:
    State transitions follow Markov chain with transition matrix P
    Observations are state-dependent with emission probabilities
    
States (Regimes):
    1. Trending (Momentum): High directional move, low volatility
    2. Mean Reverting: Oscillation around mean
    3. High Volatility: Large price swings, crisis mode
    
Applications:
    - Regime-aware strategy selection
    - Dynamic position sizing
    - Risk management (reduce exposure in high-vol regime)

References:
    Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from src.base import AssetClass


class MarketRegimeHMM:
    """
    Hidden Markov Model for identifying market regimes.
    
    Automatically detects regime switches and provides regime probabilities.
    """
    
    def __init__(self, n_regimes: int = 3, name: str = "Market_Regime_HMM"):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden states (typically 2-4)
            name: Model identifier
        """
        self.n_regimes = n_regimes
        self.name = name
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted: bool = False
        self.regime_names: Dict[int, str] = {}
        
    def fit(self, data: pd.DataFrame, features: List[str] = None) -> 'MarketRegimeHMM':
        """
        Fit HMM to market data.
        
        Args:
            data: DataFrame with OHLC and features
            features: List of feature columns to use (default: returns, volatility, volume)
            
        Returns:
            self (for chaining)
        """
        if features is None:
            # Default features for regime detection
            features = []
            if 'returns' in data.columns:
                features.append('returns')
            if 'realized_vol' in data.columns:
                features.append('realized_vol')
            if 'hl_range' in data.columns:
                features.append('hl_range')
        
        if len(features) == 0:
            raise ValueError("No valid features found in data")
        
        # Prepare feature matrix
        X = data[features].dropna().values
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for HMM fitting: {len(X)} observations")
        
        # Standardize features
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0)
        X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(X_normalized)
        self.is_fitted = True
        
        # Identify regimes based on characteristics
        self._label_regimes(data, features)
        
        print(f"HMM fitted with {self.n_regimes} regimes")
        print(f"Log-likelihood: {self.model.score(X_normalized):.2f}")
        print("\nRegime Characteristics:")
        for regime_id, regime_name in self.regime_names.items():
            print(f"  Regime {regime_id}: {regime_name}")
        
        return self
    
    def _label_regimes(self, data: pd.DataFrame, features: List[str]):
        """
        Automatically label regimes based on their characteristics.
        
        Uses cluster means to identify:
        - High volatility regime
        - Trending regime (high returns)
        - Mean reverting regime (low returns, low vol)
        """
        X = data[features].dropna().values
        X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Get states
        states = self.model.predict(X_normalized)
        
        # Analyze each regime
        regime_stats = []
        for i in range(self.n_regimes):
            mask = states == i
            
            if 'returns' in features:
                mean_return = data.loc[data[features].dropna().index, 'returns'][mask].mean()
            else:
                mean_return = 0
            
            if 'realized_vol' in features:
                mean_vol = data.loc[data[features].dropna().index, 'realized_vol'][mask].mean()
            else:
                mean_vol = 0
            
            regime_stats.append({
                'id': i,
                'mean_return': mean_return,
                'mean_vol': mean_vol,
                'frequency': mask.sum() / len(states)
            })
        
        # Sort by volatility to assign labels
        regime_stats.sort(key=lambda x: x['mean_vol'])
        
        # Assign names based on characteristics
        for idx, stats in enumerate(regime_stats):
            regime_id = stats['id']
            
            if idx == len(regime_stats) - 1:
                # Highest volatility
                self.regime_names[regime_id] = "High_Volatility"
            elif abs(stats['mean_return']) > self.feature_stds[0] if 'returns' in features else False:
                # High absolute returns
                self.regime_names[regime_id] = "Trending"
            else:
                # Low vol, low returns
                self.regime_names[regime_id] = "Mean_Reverting"
        
    def predict_regime(self, data: pd.DataFrame, features: List[str] = None) -> np.ndarray:
        """
        Predict regime for new data.
        
        Returns:
            Array of regime labels (0 to n_regimes-1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if features is None:
            features = ['returns', 'realized_vol', 'hl_range']
        
        X = data[features].dropna().values
        X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
        
        return self.model.predict(X_normalized)
    
    def predict_proba(self, data: pd.DataFrame, features: List[str] = None) -> np.ndarray:
        """
        Get regime probabilities for each observation.
        
        Returns:
            Array of shape (n_samples, n_regimes) with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if features is None:
            features = ['returns', 'realized_vol', 'hl_range']
        
        X = data[features].dropna().values
        X_normalized = (X - self.feature_means) / (self.feature_stds + 1e-8)
        
        return self.model.predict_proba(X_normalized)
    
    def get_regime_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Returns:
            DataFrame with regime statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        features = ['returns', 'realized_vol', 'hl_range']
        available_features = [f for f in features if f in data.columns]
        
        regimes = self.predict_regime(data, features=available_features)
        
        stats = []
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            regime_data = data.loc[data[available_features].dropna().index][mask]
            
            stats.append({
                'Regime': self.regime_names.get(regime_id, f"Regime_{regime_id}"),
                'Frequency': mask.sum() / len(regimes),
                'Avg_Return': regime_data['returns'].mean() if 'returns' in regime_data else np.nan,
                'Avg_Volatility': regime_data['realized_vol'].mean() if 'realized_vol' in regime_data else np.nan,
                'Sharpe': (regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252)) 
                         if 'returns' in regime_data and regime_data['returns'].std() > 0 else np.nan
            })
        
        return pd.DataFrame(stats)
