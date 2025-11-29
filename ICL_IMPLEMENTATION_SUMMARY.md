# In-Context Learning Implementation Summary

## Overview

The system now learns from past effective notifications and uses those examples to improve future decisions. When the LLM decides whether to send a notification, it can see examples of similar situations where notifications worked well, helping it make better choices over time.

## Key Changes

The core addition is an in-context learning system that tracks which notifications were effective and reuses that knowledge. When making a decision, the system looks for past effective notifications that match the current situation based on timing and goal alignment. These examples are formatted and added to the LLM prompt, so the model can learn from what worked before.

Decision logs now include additional fields for tracking this learning process. Each decision records things like how long it's been since the last notification, which goal bucket the situation falls into, how many effective examples were available, and which specific examples were used in making the decision. This creates a feedback loop where effective interventions become examples for future decisions.

The system uses a bucket-based matching approach. Time since last notification is grouped into buckets like "0-60 seconds" or "300+ seconds", and goal relevance is grouped into "high", "medium", "low", or "none". When selecting examples, the system finds past effective notifications that match both the current time bucket and goal bucket, ensuring the examples are relevant to the current context.

## Modified Files

The main changes are in `gum/notifier.py`, which now includes example selection logic, time and goal bucket calculations, and integration of examples into the decision-making prompt. The prompt template in `gum/prompts/gum.py` was updated to include a learning examples section that appears when examples are available. The observation window system in `gum/adaptive_nudge/observation_window.py` was enhanced to track effectiveness scores that feed into the learning system.

## How It Works

When a notification is sent, the system starts a two-minute observation window to evaluate whether the notification was effective. If the effectiveness score is high enough, that decision becomes an example that can be used in future similar situations. Over time, as more effective examples accumulate, the system gets better at selecting relevant examples and making decisions that align with what has worked in the past.

The learning happens automatically as the system runs. Early decisions have no examples available, so the system operates in a "cold start" mode. As effective notifications accumulate, examples become available and start being used in prompts. The system tracks which examples were used for each decision, creating a clear audit trail of how learning influenced each choice.

## New Helper Functions and Configuration

The implementation adds nine new helper functions to support the learning system. `_get_time_bucket()` calculates which time bucket a situation falls into based on seconds since the last notification, returning values like "0-60s" or "300s+". `_get_goal_alignment_bucket()` converts goal relevance scores into buckets of "high", "medium", "low", or "none". `_calculate_time_since_last_nudge()` computes the time elapsed since the last notification, returning infinity if no previous notification exists. `_calculate_goal_alignment()` normalizes goal relevance scores from the 0-10 scale to a 0-1 scale. `_calculate_frequency_context()` counts how many notifications were sent in the last hour to provide context about recent activity.

The system includes two example selection functions. `_select_examples_any_goal()` performs preliminary selection matching only on time bucket before the goal bucket is known, giving the LLM some context even during cold start. `_select_examples_from_log()` performs full selection matching on both goal and time buckets after the decision is made, used for accurate tracking. Both functions filter for effective examples with scores of 0.7 or higher and return up to five most recent matches.

Additional helper functions include `_count_effective_examples()` which counts how many effective examples are available in the decision log, `_generate_observation_pattern_summary()` which creates brief pattern descriptions for matching, and `_format_examples_for_prompt()` which converts example dictionaries into formatted text for inclusion in the LLM prompt.

The system uses a policy version field set to "v1.0" by default, which can be manually updated when policy changes are made. This allows tracking which policy version was used for each decision, enabling analysis of how policy changes affect outcomes. The system also includes a feature flag `in_context_learning_enabled` that can disable in-context learning if needed, though it's enabled by default.

Threshold logic is built into several key functions. The effectiveness threshold for considering a notification as an effective example is set at 0.7, meaning only notifications with effectiveness scores of 0.7 or higher become examples. Time buckets are divided into ranges: 0-60 seconds, 60-120 seconds, 120-180 seconds, 180-300 seconds, and 300+ seconds. Goal alignment buckets are determined by relevance scores, with high being 7 or above, medium being 4-6, and low being below 4. The system limits example selection to the top 5 most recent matching examples, and the frequency context window looks back one hour. All of these thresholds are marked with `#hyperparameter, can tune` comments in the code, making it easy to adjust them for different use cases or research experiments.

## Testing and Verification

To verify in-context learning is working, check the decision log file at `notification_decisions_{username}.json`. Each decision contains ICL fields showing how learning influenced the decision.

**Key ICL fields to check:**
- `policy_version`: Policy version used (default "v1.0") - gets better over time! Manually change this when substantive backend changes happen
- `examples_available_count`: Number of effective examples available at decision time
- `examples_used_count`: Number of examples actually included in the prompt
- `examples_used`: List of nudge IDs of examples that influenced this decision
- `goal_alignment_bucket`: Goal relevance bucket ("high", "medium", "low", "none")
- `time_since_last_nudge_bucket`: Time bucket ("0-60s", "60-120s", etc.)

**Run tests:**
```bash
# Run ICL unit tests
pytest tests/test_in_context_learning.py -v

# Run specific test classes
pytest tests/test_in_context_learning.py::TestExampleSelectionIntegration -v
pytest tests/test_in_context_learning.py::TestDecisionSaving -v
```

**Monitor in real-time:**
While GUM is running, use a separate terminal to monitor ICL activity:
```bash
python3 monitor_icl_realtime.py username
```
This shows new decisions as they're created with all ICL fields displayed.

