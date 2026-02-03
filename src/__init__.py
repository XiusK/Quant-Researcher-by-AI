"""
Quant Research Framework - Core Module

A rigorous quantitative trading framework emphasizing:
- Mathematical correctness over intuition
- Strict risk management
- Bias detection and prevention

Author: Quant Researcher AI
"""

from src.base import (
    BaseStochasticModel,
    BaseStrategy,
    BaseRiskManager,
    AssetClass,
    ModelParameters,
    Signal,
    RiskMetrics,
    run_stationarity_tests,
    check_outliers
)

__version__ = "0.1.0"
__all__ = [
    "BaseStochasticModel",
    "BaseStrategy", 
    "BaseRiskManager",
    "AssetClass",
    "ModelParameters",
    "Signal",
    "RiskMetrics",
    "run_stationarity_tests",
    "check_outliers"
]
