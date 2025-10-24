# Adaptive Nudge Engine

## Overview
System for learning personalized notification policies through implicit feedback loops. Captures user behavior after nudges and evaluates effectiveness using LLM judgment.

## Goal
Replace explicit feedback buttons with implicit feedback by observing user actions after nudge delivery.

## Implementation

### Core Modules (`gum/adaptive_nudge/`)
- `state_capture.py` - Captures system state (apps, tabs, files, clipboard)
- `observation_window.py` - Manages 3-minute observation periods
- `llm_judge.py` - Evaluates nudge effectiveness using LLM
- `training_logger.py` - Logs training data in JSONL format

### Integration
- Modified `gum/notifier.py` to trigger observation windows after notifications
- Uses Google AI Studio API with `gemini-2.5-flash` model
- Graceful error handling for macOS privacy restrictions

## Execution Flow
```
1. GUM sends nudge → 2. Start 3-min observation → 3. User acts/ignores → 
4. Capture system state → 5. LLM evaluates effectiveness → 6. Log training data
```

## Usage
```bash
# Start GUM with adaptive nudge engine
python -m gum.cli -u "Your Name" --enable-notifications --model "gemini-1.5-pro"
```

## Requirements
- macOS accessibility permissions for screen capture
- Google AI Studio API key
- GUM system running with notifications enabled