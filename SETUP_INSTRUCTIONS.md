# Setup Instructions for CustomNudge Adaptive Nudge Engine

## Prerequisites
- Python 3.11+ installed
- macOS with Accessibility Permissions granted to Terminal and Python
- Google AI Studio API key: `AIzaSyBKYsQ77txAwduIif62yQPanVx2QYG7CuE`

## Installation Steps

### 1. Navigate to Project Directory
```bash
cd /path/to/customNudge
```

### 2. Install Dependencies
```bash
pip install pillow mss pynput shapely pyobjc-framework-Quartz openai SQLAlchemy pydantic sqlalchemy-utils python-dotenv scikit-learn aiosqlite greenlet persist-queue
```

### 3. Set API Key
```bash
export GOOGLE_API_KEY="AIzaSyBKYsQ77txAwduIif62yQPanVx2QYG7CuE"
```

**Note:** You must set this environment variable in the same shell session where you run the GUM system, or include it directly in the command (see step 4).

### 4. Start GUM System
```bash
export GOOGLE_API_KEY="AIzaSyBKYsQ77txAwduIif62yQPanVx2QYG7CuE" && python -m gum.cli -u "Eyrin" --enable-notifications --model "gemini-2.5-flash" &
```

**Model Selection:** The system uses `gemini-2.5-flash` for all LLM operations (proposition generation, nudge decisions, and effectiveness evaluation).

### 5. Start Notification GUI
```bash
python notification_gui.py Eyrin &
```

This opens a GUI window that displays notification decisions in real-time.

## Configuration Options

### Batch Size Configuration
You can adjust batch processing parameters:
```bash
python -m gum.cli -u "Eyrin" --enable-notifications --model "gemini-2.5-flash" --min-batch-size 3 --max-batch-size 10
```

- `--min-batch-size`: Minimum observations before processing (default: 5)
- `--max-batch-size`: Maximum observations per batch (default: 50)

### macOS Permissions
The system requires Accessibility permissions to capture window titles. If you see errors like:
```
execution error: System Events got an error: Can't get frontmost of window. (-1728)
```

This is handled gracefully—the system will continue without window titles. To enable full functionality:
1. Go to System Settings → Privacy & Security → Accessibility
2. Grant permissions to Terminal and Python

## Current Known Issue

**Batch Processing Loop Not Running:** The system collects observations in the queue (you can verify this by checking queue size in logs), but the batch processing loop that triggers proposition generation and nudge decisions is not executing reliably. This prevents the system from:
- Generating propositions from observations
- Making nudge decisions
- Triggering the adaptive nudge feedback loop

**Workaround:** The queue accumulates observations correctly, but manual intervention may be needed to trigger batch processing. We are actively debugging this issue.

## Verification

### Check System Status
```bash
# Check if processes are running
ps aux | grep "gum.cli"
ps aux | grep "notification_gui"

# Check observation queue size (in logs)
tail -f ~/.cache/gum/logs/gum.log | grep "queue size"
```

### Check Training Data Collection
```bash
# View collected training data
cat ~/.cache/gum/training_data_eyrin.jsonl
```

### Check Notification Decisions
```bash
# View notification contexts
cat notification_contexts_eyrin.json

# View notification decisions
cat notification_decisions_eyrin.json
```

## Troubleshooting

### API Key Not Working
- Ensure `GOOGLE_API_KEY` is set in the same shell session
- Check logs for API key errors: `tail -f ~/.cache/gum/logs/gum.log | grep -i "api key"`

### Queue Corruption
If you see `EOFError: Ran out of input`:
```bash
# Clear corrupted queue
rm -rf ~/.cache/gum/batches/queue/*
```

### No Notifications Appearing
1. Check that batch processing is running (see "Current Known Issue" above)
2. Verify observations are being collected: check queue size in logs
3. Wait for minimum batch size to be reached before processing begins
4. Check notification GUI window for decision logs

## Data Locations

- **Training Data:** `~/.cache/gum/training_data_eyrin.jsonl`
- **Notification Contexts:** `notification_contexts_eyrin.json`
- **Notification Decisions:** `notification_decisions_eyrin.json`
- **Database:** `~/.cache/gum/gum.db`
- **Logs:** `~/.cache/gum/logs/gum.log`
- **Queue:** `~/.cache/gum/batches/queue/`


