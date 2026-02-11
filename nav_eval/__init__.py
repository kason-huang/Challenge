# ruff: noqa
"""
Navigation Task Evaluation System

A modular, extensible evaluation framework for navigation tasks, inspired by Habitat.
Provides decoupled architecture with pluggable benchmarks, standard interfaces,
and high-performance parallel evaluation.
"""

__version__ = "0.1.0"
__author__ = "Navigation Research Team"

# Core interfaces and classes
from nav_eval.core.interfaces import Agent, Simulator
from nav_eval.core.environment import EvaluationEnv
from nav_eval.core.metrics import MetricsCalculator
from nav_eval.benchmarks.registry import BenchmarkRegistry

# Configuration
from nav_eval.config.config_loader import ConfigLoader, load_config

# Vector environment for parallel evaluation
from nav_eval.env.vector_env import VectorEnv, make_vector_env

__all__ = [
    "Agent",
    "Simulator",
    "EvaluationEnv",
    "MetricsCalculator",
    "BenchmarkRegistry",
    "ConfigLoader",
    "load_config",
    "VectorEnv",
    "make_vector_env",
]
