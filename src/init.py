"""
Paper Review Attack Detection Package
====================================

A comprehensive toolkit for detecting prompt injection attacks in academic papers
submitted to AI-assisted peer review systems.

Modules:
--------
- data_collector: Collect papers from arXiv and other sources
- attack_generator: Generate various types of prompt injection attacks
- detector: Multi-layered detection system for identifying attacks
- evaluator: Performance evaluation and metrics calculation
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .data_collector import ArxivDatasetCollector
from .attack_generator import AttackSampleGenerator
from .detector import PromptInjectionDetector
from .evaluator import ExperimentEvaluator
from .utils import setup_logging, load_config

__all__ = [
    'ArxivDatasetCollector',
    'AttackSampleGenerator', 
    'PromptInjectionDetector',
    'ExperimentEvaluator',
    'setup_logging',
    'load_config'
]