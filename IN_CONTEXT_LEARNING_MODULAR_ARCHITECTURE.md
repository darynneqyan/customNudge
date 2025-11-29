# In-Context Learning - Modular Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GUMNotifier Class                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Feature Flag: in_context_learning_enabled (bool)   │  │
│  │  Policy Version: policy_version (str)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MODULE 1: Helper Methods (9 methods)                │  │
│  │  - Pure functions, no side effects                    │  │
│  │  - Can be deleted without breaking existing code     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MODULE 2: Enhanced Decision Logging                 │  │
│  │  - _save_decision() enhanced (conditional)            │  │
│  │  - Adds new fields only if feature enabled            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MODULE 3: Example Selection                         │  │
│  │  - _make_notification_decision() enhanced             │  │
│  │  - Returns empty list if feature disabled             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Existing Methods (UNCHANGED)                        │  │
│  │  - All existing logic preserved                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
┌─────────────────┐
│ Feature Flag    │──┐
└─────────────────┘  │
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│ Helper Methods  │      │ Decision Logging│
│ (Module 1)      │─────▶│ (Module 2)     │
└─────────────────┘      └─────────────────┘
        │                         │
        │                         │
        ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│ Example Select  │      │ Prompt Enhance  │
│ (Module 3)      │─────▶│ (Module 4)      │
└─────────────────┘      └─────────────────┘
```

## Data Flow

```
Observation Batch
    │
    ▼
process_observation_batch()
    │
    ▼
_make_notification_decision()
    │
    ├─▶ [IF ENABLED] _select_examples_from_log()
    │   │
    │   └─▶ Returns: List of example decisions
    │
    ├─▶ [IF ENABLED] _format_examples_for_prompt()
    │   │
    │   └─▶ Returns: Formatted string (or empty)
    │
    ├─▶ Build prompt (with or without examples)
    │
    ├─▶ Call LLM
    │
    └─▶ Return decision + examples_used
        │
        ▼
_save_decision()
    │
    ├─▶ [IF ENABLED] Calculate new fields using helpers
    │   │
    │   ├─▶ _calculate_time_since_last_nudge()
    │   ├─▶ _get_time_bucket()
    │   ├─▶ _calculate_frequency_context()
    │   ├─▶ _calculate_goal_alignment()
    │   ├─▶ _get_goal_alignment_bucket()
    │   ├─▶ _generate_observation_pattern_summary()
    │   ├─▶ _count_effective_examples()
    │   └─▶ Add examples_used from selection
    │
    └─▶ Save decision entry (with or without new fields)
```

## Modular Removal Guide

### To Disable Feature (Keep Code)
```python
# In __init__()
self.in_context_learning_enabled = False
```
**Result**: All new features disabled, code remains for future use

### To Remove Module 1 (Helper Methods)
**Delete**: 9 helper methods
**Impact**: None (only used if feature enabled)

### To Remove Module 2 (Enhanced Logging)
**Delete**: New field additions in `_save_decision()`
**Keep**: Existing field logging
**Impact**: None (existing fields still logged)

### To Remove Module 3 (Example Selection)
**Delete**: Example selection call in `_make_notification_decision()`
**Delete**: `examples_used` tracking
**Impact**: None (prompt works without examples)

### To Remove Module 4 (Prompt Enhancement)
**Delete**: `{learning_examples}` parameter
**Delete**: "Learning Examples" section
**Impact**: None (prompt works without examples section)

## Code Isolation Map

```
gum/notifier.py
├── __init__()
│   ├── [NEW] self.in_context_learning_enabled = True
│   └── [NEW] self.policy_version = "v1.0"
│
├── _save_decision() [MODIFIED]
│   ├── [EXISTING] All existing field logging
│   └── [NEW] if self.in_context_learning_enabled:
│       └── [NEW] Add new fields using helpers
│
├── _make_notification_decision() [MODIFIED]
│   ├── [EXISTING] All existing decision logic
│   └── [NEW] if self.in_context_learning_enabled:
│       ├── [NEW] Select examples
│       ├── [NEW] Format examples
│       └── [NEW] Add to prompt
│
├── [NEW] _get_goal_alignment_bucket() [ISOLATED]
├── [NEW] _get_time_bucket() [ISOLATED]
├── [NEW] _calculate_goal_alignment() [ISOLATED]
├── [NEW] _calculate_time_since_last_nudge() [ISOLATED]
├── [NEW] _calculate_frequency_context() [ISOLATED]
├── [NEW] _generate_observation_pattern_summary() [ISOLATED]
├── [NEW] _count_effective_examples() [ISOLATED]
├── [NEW] _select_examples_from_log() [ISOLATED]
└── [NEW] _format_examples_for_prompt() [ISOLATED]

gum/prompts/gum.py
└── NOTIFICATION_DECISION_PROMPT [MODIFIED]
    ├── [EXISTING] All existing prompt sections
    └── [NEW] {learning_examples} section (conditional)

gum/adaptive_nudge/observation_window.py
└── _complete_observation_after_delay() [MODIFIED]
    ├── [EXISTING] All existing training entry creation
    └── [NEW] Add new fields if available (using .get())
```

## Safety Guarantees

1. **Existing Code Unchanged**: All existing methods work exactly as before
2. **Backward Compatible**: Old decision logs still readable (new fields optional)
3. **Graceful Degradation**: Works with empty example pool (cold start)
4. **Feature Flag**: Can disable without code changes
5. **Isolated Modules**: Each module can be removed independently
6. **Safe Defaults**: All new fields use `.get()` with defaults

## Testing Checklist

- [ ] Cold start: No examples, all fields calculated correctly
- [ ] With examples: Examples selected and formatted correctly
- [ ] Feature disabled: No examples, prompt works normally
- [ ] Old decision logs: Can read without new fields
- [ ] Helper methods: All return correct values
- [ ] Example selection: Matches correct buckets
- [ ] Decision logging: All new fields saved correctly
- [ ] Training logger: New fields included if available
- [ ] Prompt: Works with and without examples section

