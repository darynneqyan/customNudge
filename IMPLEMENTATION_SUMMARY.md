# Adaptive Nudge Engine - Implementation Summary

## Branch: `eyrin`

## Overview
Complete adaptive nudge system with batch processing, native notifications, and learning capabilities. System monitors user behavior, generates intelligent nudges, and learns effectiveness through implicit feedback.

## Core Architecture

### Batch Processing System
- **ObservationBatcher**: Persistent SQLite queue with configurable batch sizes (min=3, max=10)
- **Processing Loop**: Event-driven batch processing triggered when queue reaches minimum size
- **API Optimization**: Reduces API calls by processing multiple observations together

### Notification Engine
- **LLM Decision Making**: Uses gemini-2.5-flash for intelligent notification decisions
- **Native macOS Notifications**: Displays actual system notifications with custom titles/icons
- **Context Analysis**: BM25 similarity search across propositions and observations
- **Adaptive Learning**: 3-minute observation windows to evaluate nudge effectiveness

### Adaptive Nudge Components
- **SystemStateCapture**: macOS AppleScript integration for app/tab/file detection
- **LLMJudge**: Effectiveness evaluation with 0/1 scoring and reasoning
- **TrainingDataLogger**: JSONL format for future ML policy development
- **ObservationWindowManager**: Async 3-minute post-nudge monitoring

## Execution Flow
1. Screen Observer captures activity every few seconds
2. Observations accumulate in persistent queue until batch size reached
3. Batch processing generates propositions using gemini-2.5-flash
4. Notification engine analyzes context and makes LLM-based decisions
5. Native macOS notifications display with appropriate titles/icons
6. Adaptive nudge engine starts 3-minute observation window
7. System state captured and LLM evaluates nudge effectiveness
8. Training data logged for continuous learning

## API Integration
- **Primary Model**: gemini-2.5-flash for all proposition generation and decisions
- **Judge Model**: gemini-2.5-flash for effectiveness evaluation
- **API Key**: GOOGLE_API_KEY environment variable
- **Provider**: GeminiProvider handles all API calls with error handling

## Configuration
```bash
export GOOGLE_API_KEY="your-api-key"
python -m gum.cli -u "User Name" --enable-notifications --model "gemini-2.5-flash" --min-batch-size 3 --max-batch-size 10
```

## Outputs
- **Native Notifications**: macOS system notifications with contextual titles
- **Training Data**: JSONL logs of nudge effectiveness for ML development
- **Decision Logs**: JSON files tracking notification decisions and reasoning
- **Context Data**: Detailed observation and proposition analysis

## Status
Fully operational system with batch processing, native notifications, and adaptive learning. System successfully processes observations, generates intelligent nudges, displays notifications, and learns from user responses.

## Files Modified
- **gum/batcher.py**: Fixed path expansion and race condition issues
- **gum/gum.py**: Corrected initialization order and standardized API calls
- **gum/notifier.py**: Added native notification display and adaptive nudge integration
- **gum/cli.py**: Standardized model to gemini-2.5-flash
- **gum/adaptive_nudge/llm_judge.py**: Standardized API configuration