# CustomNudge: Setup Instructions

## Prerequisites
- Python 3.11+ installed
- macOS with Accessibility Permissions granted to Terminal and Python Launcher
- Google AI studio API key

## Installation Steps

### 1. Navigate to Project Directory
```bash
cd /path/to/customNudge
```

### 2. Install Dependencies
```bash
pip install pillow mss pynput shapely pyobjc-framework-Quartz openai SQLAlchemy pydantic sqlalchemy-utils python-dotenv scikit-learn aiosqlite greenlet persist-queue google-generativeai>=0.8.3
```

### 3. Build Swift Notifier
The notification system requires a Swift binary to be built first:
```bash
cd notifier/swift_notifier
./build.sh
cd ../..
```

This creates the `GUM Notifier.app` bundle that handles interactive notifications.

### 4. Configure macOS System Settings

**⚠️ IMPORTANT: Do this BEFORE running the GUM system for the first time.**

#### A. Notification Permissions

The GUM Notifier app needs permission to display notifications:

1. **First, trigger the permission request** by running the notifier once:
   ```bash
   cd notifier/swift_notifier
   ./mac_notifier "Test" "Permission check"
   cd ../..
   ```

2. **Grant notification permissions:**
   - Open **System Settings** (or **System Preferences** on older macOS)
   - Navigate to **Notifications & Focus** (or **Notifications** on older macOS)
   - Find **"GUM Notifier"** in the list of apps
   - Enable **"Allow Notifications"**
   - Set **Alert Style** to **"Alerts"** 
   - Enable **"Show in Notification Center"** and **"Play sound for notifications"**
   - Turn off Focus Mode (e.g. Do Not Disturb) or add "GUM Notifier" to Allowed Apps for Notifications so that the Notifications can come through.

3. **Verify permissions:**
   ```bash
   cd notifier/swift_notifier
   ./mac_notifier "Test" "Does this work?"
   cd ../..
   ```
   If you see a notification with "Thanks!" and "Not now" buttons, permissions are correctly configured.

#### B. Accessibility Permissions (Optional but Recommended)

For full functionality including window title capture:

1. Open **System Settings** → **Privacy & Security** → **Accessibility**
2. Enable permissions for:
   - **Terminal** (or your terminal app) (Go to applications --> utilities --> terminal)
   - **Python Launcher** (Go to your python folder and click rocketship emoji Python Launcher)
3. Open **System Settings** → **Privacy & Security** → **Screen Recording** → enable the system you're running customNudge command line from (like VSCode or Cursor)

**Note:** The system will work without Accessibility permissions, but window titles won't be captured.

### 5. Set API Key
```bash
export GOOGLE_API_KEY="INSERT API KEY HERE"
```

**Note:** You must set this environment variable in the same shell session where you run the GUM system, or include it directly in the command (see step 6).

### 6. Start GUM System
```bash
export GOOGLE_API_KEY="" && python -m gum.cli -u "NAME" --enable-notifications --model "gemini-2.5-flash"
```

**Model Selection:** The system uses `gemini-2.5-flash` for all LLM operations (proposition generation, nudge decisions, and effectiveness evaluation).

## For Developers

### 7. Start Notification GUI in a Separate Terminal
```bash
python notification_gui.py NAME
```

This opens a GUI window that displays notification decisions in real-time.

## Configuration Options

### Batch Size Configuration
You can adjust batch processing parameters:
```bash
python -m gum.cli -u "NAME" --enable-notifications --model "gemini-2.5-flash" --min-batch-size 3 --max-batch-size 10
```

- `--min-batch-size`: Minimum observations before processing (default: 5)
- `--max-batch-size`: Maximum observations per batch (default: 50)

### macOS Permissions

#### Notification Permissions (Required)
The GUM Notifier requires notification permissions to display interactive nudges. See **Step 4** in the Installation Steps above for detailed instructions.

**Quick Setup:**
1. Run the notifier once to trigger permission request: `./notifier/swift_notifier/mac_notifier "Test" "Permission check"`
2. Go to **System Settings** → **Notifications & Focus**
3. Find **"GUM Notifier"** and enable **"Allow Notifications"**
4. Set **Alert Style** to **"Banners"** or **"Alerts"**

#### Accessibility Permissions (Optional)
The system requires Accessibility permissions to capture window titles. If you see errors like:
```
execution error: System Events got an error: Can't get frontmost of window. (-1728)
```

This is handled gracefully—the system will continue without window titles. To enable full functionality:
1. Go to **System Settings** → **Privacy & Security** → **Accessibility**
2. Grant permissions to **Terminal** and **Python Launcher**

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

<!-- ## Troubleshooting

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
1. **Verify notification permissions are granted:**
   - Go to **System Settings** → **Notifications & Focus**
   - Ensure **"GUM Notifier"** has **"Allow Notifications"** enabled
   - Test manually: `./notifier/swift_notifier/mac_notifier "Test" "Does this work?"`
   
2. **Check that batch processing is running** (see "Current Known Issue" above)
3. **Verify observations are being collected:** check queue size in logs
4. **Wait for minimum batch size to be reached** before processing begins
5. **Check notification GUI window** for decision logs
6. **Check logs for Swift notifier errors:**
   ```bash
   tail -f ~/.cache/gum/logs/gum.log | grep -i "swift\|notif\|fallback"
   ```

### Notification Permission Errors
If you see errors like `"Notifications denied"` or `"Authorization error"`:
1. Rebuild the Swift notifier: `cd notifier/swift_notifier && ./build.sh`
2. Grant permissions in **System Settings** → **Notifications & Focus** → **GUM Notifier**
3. Test the notifier directly: `./notifier/swift_notifier/mac_notifier "Test" "Permission check"`
4. If permissions still don't work, try removing and re-adding the app:
   - Remove from System Settings
   - Rebuild: `cd notifier/swift_notifier && ./build.sh`
   - Run again to trigger new permission request

## Data Locations

- **Training Data:** `~/.cache/gum/training_data_eyrin.jsonl`
- **Notification Contexts:** `notification_contexts_eyrin.json`
- **Notification Decisions:** `notification_decisions_eyrin.json`
- **Database:** `~/.cache/gum/gum.db`
- **Logs:** `~/.cache/gum/logs/gum.log`
- **Queue:** `~/.cache/gum/batches/queue/` -->


