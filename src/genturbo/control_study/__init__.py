"""Strict runner for the Ehrlich paper-control experiment."""

from .config import ARMS, STUDY, ControlStudyConfig, ControlStudyConfigError, load_control_study

__all__ = ["ARMS", "STUDY", "ControlStudyConfig", "ControlStudyConfigError", "load_control_study"]
