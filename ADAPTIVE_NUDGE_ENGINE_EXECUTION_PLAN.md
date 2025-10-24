# Adaptive Nudge Engine - Implementation Plan & Key Changes

## Overview

This document outlines the implementation of the Adaptive Nudge Engine, a system for learning personalized notification policies through implicit feedback loops. The engine extends the existing GUM (General User Models) system to capture user behavior after nudges and evaluate their effectiveness using LLM-based judgment.

## Goals & Research Challenge

**Primary Goal**: Move away from explicit feedback buttons ("Got it!" / "Not now") to an implicit feedback loop based on observing user actions after nudge delivery.

**Research Challenge**: Develop a system that can learn when to send helpful notifications by observing user behavior patterns and evaluating nudge effectiveness automatically.

## Key Changes Implemented

### 1. New Module Structure (`gum/adaptive_nudge/`)

**Purpose**: Modular design for easy testing and maintenance of adaptive nudge components.

**Files Created**:
- `__init__.py` - Module initialization and exports
- `state_capture.py` - System state capture functionality
- `observation_window.py` - 3-minute observation window management
- `llm_judge.py` - LLM-based effectiveness evaluation
- `training_logger.py` - Structured training data logging

### 2. System State Capture (`state_capture.py`)

**Goal**: Replace visual screenshots with structured text-based system state representation.

**Key Features**:
- Captures active application and window title
- Extracts browser tabs from Safari, Chrome, Firefox
- Records clipboard history and open files
- Monitors recent applications and system info
- Uses macOS AppleScript for reliable data extraction

**Integration Point**: Called automatically after 3-minute observation window

### 3. Observation Window Management (`observation_window.py`)

**Goal**: Implement asynchronous 3-minute observation periods without blocking the main GUM system.

**Key Features**:
- **Duration Changed**: 3 minutes (reduced from 5 minutes as requested)
- Asynchronous task management using `asyncio`
- Automatic state capture after observation period
- LLM evaluation of captured state
- Training data logging for future policy development

**Integration Point**: Triggered automatically when notifications are sent

### 4. LLM Judge System (`llm_judge.py`)

**Goal**: Provide impartial, consistent evaluation of nudge effectiveness using specialized LLM prompts.

**Key Features**:
- High-quality prompt designed for behavioral psychology evaluation
- Conservative scoring (only score "effective" with clear evidence)
- Structured reasoning extraction
- Batch evaluation capabilities
- Error handling with default to "ineffective" on errors

**Integration Point**: Called after system state capture to evaluate effectiveness

### 5. Training Data Logger (`training_logger.py`)

**Goal**: Create comprehensive dataset for machine learning model development.

**Key Features**:
- JSONL format for easy ML processing
- Structured data with nudge context, system state, and effectiveness scores
- Statistics and analytics capabilities
- Export functionality for model training
- Recent entries filtering

**Integration Point**: Logs all nudge effectiveness data for future policy learning

### 6. GUM Notifier Integration (`gum/notifier.py`)

**Goal**: Seamlessly integrate adaptive nudge engine with existing notification system.

**Key Changes**:
- Added `ObservationWindowManager` initialization
- Integrated observation window triggering after notification sending
- Added adaptive nudge statistics and management methods
- Maintained backward compatibility with existing notification system

**Integration Point**: Modified notification sending logic to start observation windows

## Execution Flow

```
1. GUM detects situation → Sends nudge (existing logic)
2. Adaptive Nudge Engine starts 3-minute observation window
3. User acts/ignores nudge during observation period
4. System captures post-nudge state (structured data)
5. LLM Judge evaluates effectiveness (0 or 1)
6. Training data logged to JSONL file
7. Future policy model can use this data for learning
```

## Technical Implementation Details

### Asynchronous Architecture

**Why**: Prevents blocking the main GUM system during observation periods.

**Implementation**: 
- Uses `asyncio.create_task()` for non-blocking observation windows
- Each observation runs independently
- Automatic cleanup after completion

### System State Capture

**Why**: Structured data is more reliable than visual analysis for behavioral patterns.

**Implementation**:
- macOS AppleScript for application and window detection
- Browser tab extraction from multiple browsers
- File system monitoring for open files
- Clipboard and application history tracking

### LLM Judge Prompt Design

**Why**: Need impartial, consistent evaluation of user behavior.

**Implementation**:
- Specialized prompt for human-computer interaction evaluation
- Conservative scoring to avoid false positives
- Structured reasoning extraction
- Context-aware evaluation considering application usage

### Training Data Structure

**Why**: Need structured data for machine learning model development.

**Implementation**:
- JSONL format for easy processing
- Complete nudge context and user state
- LLM reasoning and effectiveness scores
- Timestamp and metadata for analysis

## Configuration & Usage

### Environment Variables

```bash
# Enable adaptive nudge engine
export ADAPTIVE_NUDGE_ENABLED="true"

# LLM Judge configuration
export JUDGE_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="your-api-key"

# Observation duration (seconds)
export OBSERVATION_DURATION="180"  # 3 minutes
```

### Running the System

```bash
# Start GUM with adaptive nudge engine
python -m gum.cli -u "Your Name" --enable-notifications

# The system will now:
# 1. Send nudges as before
# 2. Start 3-minute observation windows
# 3. Capture system state after observation
# 4. Get LLM judgment on effectiveness
# 5. Log training data for policy model development
```

## Data Outputs

### Training Data File
- Location: `~/.cache/gum/training_data_{user_name}.jsonl`
- Format: One JSON object per line
- Content: Complete nudge effectiveness data

### Statistics Available
- Active observation count
- Training data effectiveness rates
- Nudge type distribution
- Recent activity analysis

## Benefits & Research Value

### For Users
- **Implicit Feedback**: No need for explicit "Got it!" buttons
- **Learning System**: Gets better at timing nudges over time
- **Privacy**: No visual screenshots, only structured data

### For Researchers
- **Rich Dataset**: Comprehensive nudge effectiveness data
- **Behavioral Insights**: Understanding of what makes nudges effective
- **Policy Learning**: Foundation for ML-based nudge optimization

### For Developers
- **Modular Design**: Easy to test and modify individual components
- **Async Architecture**: Non-blocking system integration
- **Extensible**: Easy to add new state capture methods or evaluation criteria

## Future Enhancements

1. **Machine Learning Integration**: Use training data to develop better nudge policies
2. **Real-time Adaptation**: Adjust nudge timing based on effectiveness patterns
3. **Multi-modal State Capture**: Add more system state indicators
4. **A/B Testing**: Compare different nudge strategies
5. **User Preference Learning**: Adapt to individual user patterns

## Testing & Validation

### Unit Tests Needed
- System state capture accuracy
- LLM judge consistency
- Observation window timing
- Training data logging

### Integration Tests Needed
- End-to-end nudge → observation → evaluation flow
- Async task management
- Error handling and recovery

### Validation Metrics
- LLM judge inter-rater reliability
- Training data quality
- System performance impact
- User behavior pattern detection accuracy

## Conclusion

The Adaptive Nudge Engine provides a comprehensive foundation for learning personalized notification policies through implicit feedback. The modular design ensures easy testing and refinement, while the asynchronous architecture maintains system performance. The structured training data will enable future development of sophisticated ML-based nudge optimization systems.

The 3-minute observation window provides a good balance between capturing user behavior and not being too intrusive, while the LLM judge system ensures consistent, impartial evaluation of nudge effectiveness.
