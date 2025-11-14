#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_FILE="${SCRIPT_DIR}/mac_notifier.swift"
INFO_PLIST="${SCRIPT_DIR}/Info.plist"
OUTPUT_BINARY="${SCRIPT_DIR}/mac_notifier"
APP_BUNDLE="${SCRIPT_DIR}/GUM Notifier.app"
APP_CONTENTS="${APP_BUNDLE}/Contents"
APP_MACOS="${APP_CONTENTS}/MacOS"
APP_BINARY="${APP_MACOS}/GUM Notifier"

echo "🔨 Building GUM Notifier..."

# Clean previous build
rm -rf "${APP_BUNDLE}" "${OUTPUT_BINARY}"

# Step 1: Compile Swift with Cocoa framework
echo "📦 Compiling Swift source..."
swiftc -o "${OUTPUT_BINARY}" \
    "${SWIFT_FILE}" \
    -framework Foundation \
    -framework AppKit \
    -framework UserNotifications

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

chmod +x "${OUTPUT_BINARY}"
echo "✅ Binary compiled"

# Step 2: Create app bundle structure
echo "📁 Creating app bundle..."
mkdir -p "${APP_MACOS}"
cp "${OUTPUT_BINARY}" "${APP_BINARY}"
chmod +x "${APP_BINARY}"
cp "${INFO_PLIST}" "${APP_CONTENTS}/Info.plist"

# Step 3: Code sign the app bundle
echo "🔏 Code signing..."
codesign --force --deep --sign - "${APP_BUNDLE}"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Code signing failed, but continuing..."
fi

# Step 4: Create convenience symlink
ln -sf "${APP_BINARY}" "${OUTPUT_BINARY}"

echo ""
echo "✅ Build complete!"
echo "   App bundle: ${APP_BUNDLE}"
echo "   Binary: ${OUTPUT_BINARY} (symlink)"
echo ""
echo "To test:"
echo "  \"${OUTPUT_BINARY}\" \"Test Title\" \"Test Message\""
