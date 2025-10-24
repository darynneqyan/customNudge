#!/usr/bin/env python3
"""
Adaptive Nudge Engine Module

This module implements the Adaptive Nudge Engine for learning personalized
notification policies through implicit feedback loops.

Components:
- state_capture: Captures system state snapshots
- observation_window: Manages post-nudge observation periods
- llm_judge: Evaluates nudge effectiveness using LLM
- training_logger: Logs structured training data
"""

from .state_capture import SystemStateCapture
from .observation_window import ObservationWindowManager, NudgeObservation
from .llm_judge import LLMJudge
from .training_logger import TrainingDataLogger

__all__ = [
    'SystemStateCapture',
    'ObservationWindowManager', 
    'NudgeObservation',
    'LLMJudge',
    'TrainingDataLogger'
]
