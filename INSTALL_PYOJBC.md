# Installing PyObjC for NSUserNotification

## Problem
The error `NotificationDelegate is overriding existing Objective-C class` occurred because:
1. The `NotificationDelegate` class was being defined inside a method, causing redefinition errors
2. Missing PyObjC framework packages

## Solution

### Install PyObjC (macOS system Python)
```bash
python3 -m pip install --user --break-system-packages pyobjc-framework-Cocoa
```

This will automatically install:
- `pyobjc-core` (required dependency)
- `pyobjc-framework-Cocoa` (includes Foundation and AppKit)

### Alternative: Install with pip (if using virtual environment)
```bash
pip install pyobjc-framework-Cocoa
```

**Note:** `pyobjc-framework-Foundation` doesn't exist as a separate package - `Foundation` is included in `pyobjc-framework-Cocoa`.

## What Was Fixed

1. **Moved `NotificationDelegate` to module level**: The class is now defined once at the top of the file, preventing redefinition errors
2. **Added proper error handling**: The code gracefully falls back to simple notifications if PyObjC isn't available
3. **Updated dependencies**: Added `pyobjc-framework-Cocoa` and `pyobjc-framework-Foundation` to `pyproject.toml` and `setup.py`

## Verification

After installation, verify PyObjC is working:
```python
python3 -c "from Foundation import NSUserNotification; print('PyObjC Foundation installed successfully')"
```

## Note

The linter warnings about unresolved imports are expected - PyObjC is macOS-specific and may not be available in linting environments. The code handles this gracefully with try/except blocks.

