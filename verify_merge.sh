#!/bin/bash
# Quick verification script for batch processing + ICL merge

USERNAME="test"
DECISIONS_FILE="notification_decisions_${USERNAME}.json"
CONTEXTS_FILE="notification_contexts_${USERNAME}.json"

echo "=========================================="
echo "VERIFYING BATCH PROCESSING + ICL MERGE"
echo "=========================================="
echo ""

# Check if decision file exists
if [ ! -f "$DECISIONS_FILE" ]; then
    echo "⚠️  Decision file not found: $DECISIONS_FILE"
    echo "   Waiting for first batch to process..."
    echo ""
else
    echo "✅ Decision file exists: $DECISIONS_FILE"
    
    # Check if file has content
    DECISION_COUNT=$(jq '. | length' "$DECISIONS_FILE" 2>/dev/null || echo "0")
    if [ "$DECISION_COUNT" -eq 0 ]; then
        echo "⚠️  No decisions yet. Waiting for batch processing..."
    else
        echo "✅ Found $DECISION_COUNT decision(s)"
        
        # Get the most recent decision
        echo ""
        echo "--- MOST RECENT DECISION ---"
        jq '.[-1]' "$DECISIONS_FILE" 2>/dev/null | head -30
        
        echo ""
        echo "--- BATCH PROCESSING CHECK ---"
        HAS_BATCH=$(jq '.[-1] | has("batch_size")' "$DECISIONS_FILE" 2>/dev/null || echo "false")
        if [ "$HAS_BATCH" = "true" ]; then
            BATCH_SIZE=$(jq '.[-1].batch_size' "$DECISIONS_FILE" 2>/dev/null || echo "N/A")
            echo "✅ BATCH PROCESSING: Working (batch_size: $BATCH_SIZE)"
        else
            echo "❌ BATCH PROCESSING: Missing batch_size field"
        fi
        
        echo ""
        echo "--- ICL CHECK ---"
        HAS_ICL=$(jq '.[-1] | has("examples_used_count")' "$DECISIONS_FILE" 2>/dev/null || echo "false")
        if [ "$HAS_ICL" = "true" ]; then
            EXAMPLES_COUNT=$(jq '.[-1].examples_used_count' "$DECISIONS_FILE" 2>/dev/null || echo "0")
            EXAMPLES_AVAILABLE=$(jq '.[-1].examples_available_count' "$DECISIONS_FILE" 2>/dev/null || echo "0")
            GOAL_BUCKET=$(jq -r '.[-1].goal_alignment_bucket // "N/A"' "$DECISIONS_FILE" 2>/dev/null)
            TIME_BUCKET=$(jq -r '.[-1].time_since_last_nudge_bucket // "N/A"' "$DECISIONS_FILE" 2>/dev/null)
            
            echo "✅ ICL: Working"
            echo "   - Examples used: $EXAMPLES_COUNT"
            echo "   - Examples available: $EXAMPLES_AVAILABLE"
            echo "   - Goal bucket: $GOAL_BUCKET"
            echo "   - Time bucket: $TIME_BUCKET"
        else
            echo "❌ ICL: Missing ICL fields (examples_used_count, etc.)"
        fi
        
        echo ""
        echo "--- INTEGRATION CHECK ---"
        if [ "$HAS_BATCH" = "true" ] && [ "$HAS_ICL" = "true" ]; then
            echo "✅ INTEGRATION: Both batch processing AND ICL working together!"
        elif [ "$HAS_BATCH" = "true" ]; then
            echo "⚠️  INTEGRATION: Batch working but ICL fields missing"
        elif [ "$HAS_ICL" = "true" ]; then
            echo "⚠️  INTEGRATION: ICL working but batch fields missing"
        else
            echo "❌ INTEGRATION: Neither batch nor ICL fields found"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "REAL-TIME MONITORING COMMANDS:"
echo "=========================================="
echo ""
echo "1. Watch for batch processing in logs:"
echo "   tail -f ~/.cache/gum/logs/gum.log | grep -E 'batched|batch|Processing.*observations'"
echo ""
echo "2. Monitor decision file for new entries:"
echo "   watch -n 2 'jq length $DECISIONS_FILE'"
echo ""
echo "3. Watch most recent decision:"
echo "   watch -n 2 'jq .[-1] $DECISIONS_FILE'"
echo ""
echo "4. Check for ICL example selection:"
echo "   tail -f ~/.cache/gum/logs/gum.log | grep -E 'examples|ICL|in-context'"
echo ""

