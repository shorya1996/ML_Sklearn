# util/__init__.py
"""Utility package for the rule extractor app."""

from .logger import setup_logger
from .data_preprocessing import preprocess_data, safe_frequency_encode, safe_topn_collapse
from .fake_data import generate_realistic_synthetic_dataset
from .statistical_analysis import perform_statistical_analysis
from .rulefit_model import RuleFit

__all__ = [
    "setup_logger",
    "preprocess_data",
    "safe_frequency_encode",
    "safe_topn_collapse",
    "generate_realistic_synthetic_dataset",
    "perform_statistical_analysis",
    "RuleFit",
]
