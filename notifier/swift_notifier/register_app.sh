#!/bin/bash
# Script to register the GUM Notifier app with macOS so it appears in System Settings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_BUNDLE="${SCRIPT_DIR}/GUM Notifier.app"

echo "Registering GUM Notifier app with macOS..."
echo "App bundle: ${APP_BUNDLE}"
echo ""

if [ ! -d "${APP_BUNDLE}" ]; then
    echo "❌ App bundle not found! Run ./build.sh first."
    exit 1
fi

echo "Step 1: Opening app bundle to register with Launch Services..."
# Use 'open' command to register the app with macOS
open "${APP_BUNDLE}" 2>/dev/null || true

echo "Step 2: Waiting for macOS to register the app..."
sleep 2

echo "Step 3: Checking if app is registered..."
# Check if the bundle identifier is recognized
bundle_id="com.gum.notifier"
if defaults read "${APP_BUNDLE}/Contents/Info.plist" CFBundleIdentifier >/dev/null 2>&1; then
    echo "✅ App bundle identifier found: ${bundle_id}"
else
    echo "⚠️  Could not verify bundle identifier"
fi

echo ""
echo "Next steps:"
echo "1. Check System Settings → Notifications & Focus"
echo "2. Look for 'GUM Notifier' in the app list"
echo "3. If it appears, enable 'Allow Notifications'"
echo ""
echo "If 'GUM Notifier' doesn't appear yet:"
echo "- Try running the app directly: open '${APP_BUNDLE}'"
echo "- Or run a test notification: ./mac_notifier 'Test' 'Testing'"
echo "- macOS may prompt for permissions on first run"

