#!/usr/bin/env python3
"""
Test script to check macOS notification permissions for Swift notifier.
This script will:
1. Verify the Swift binary exists
2. Test if notification permissions are granted
3. Request permissions if needed
4. Display a test notification
"""

import subprocess
import sys
import time
from pathlib import Path

def get_swift_binary_path():
    """Get the path to the Swift notifier binary."""
    project_root = Path(__file__).parent.parent
    return project_root / "notifier" / "swift_notifier" / "mac_notifier"

def test_binary_exists():
    """Test 1: Check if binary exists and is executable."""
    print("=" * 70)
    print("TEST 1: Binary Existence Check")
    print("=" * 70)
    
    swift_path = get_swift_binary_path()
    print(f"Looking for binary at: {swift_path}")
    
    if not swift_path.exists():
        print("❌ FAIL: Binary not found!")
        print(f"\nTo build the binary, run:")
        print(f"  cd notifier/swift_notifier && ./build.sh")
        return False
    
    if not swift_path.is_file():
        print("❌ FAIL: Path exists but is not a file!")
        return False
    
    # Check if executable
    import os
    if not os.access(swift_path, os.X_OK):
        print("❌ FAIL: Binary exists but is not executable!")
        print(f"\nTo make it executable, run:")
        print(f"  chmod +x {swift_path}")
        return False
    
    print(f"✅ PASS: Binary exists and is executable")
    print(f"   Size: {swift_path.stat().st_size} bytes")
    return True

def test_binary_usage():
    """Test 2: Check binary shows usage message with wrong args."""
    print("\n" + "=" * 70)
    print("TEST 2: Binary Usage Message")
    print("=" * 70)
    
    swift_path = get_swift_binary_path()
    
    try:
        result = subprocess.run(
            [str(swift_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 1 and "Usage" in result.stderr:
            print("✅ PASS: Binary correctly shows usage message")
            print(f"   Error message: {result.stderr.strip()}")
            return True
        else:
            print(f"⚠️  WARNING: Unexpected response (code: {result.returncode})")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FAIL: Binary hung (should exit immediately with wrong args)")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error running binary: {e}")
        return False

def test_notification_permissions():
    """Test 3: Check notification permissions."""
    print("\n" + "=" * 70)
    print("TEST 3: Notification Permissions Check")
    print("=" * 70)
    
    swift_path = get_swift_binary_path()
    
    print("Attempting to request notification permissions...")
    print("(macOS will show a permission dialog if not already granted)")
    print()
    
    try:
        # Run with valid arguments - this will request permissions
        result = subprocess.run(
            [str(swift_path), "🧪 Permission Test", "Checking notification permissions..."],
            capture_output=True,
            text=True,
            timeout=35
        )
        
        print(f"Exit code: {result.returncode}")
        
        if result.stderr:
            print(f"Stderr output: {result.stderr.strip()}")
        
        if result.returncode == 1:
            if "permission denied" in result.stderr.lower() or "error 1" in result.stderr.lower():
                print("❌ FAIL: Notification permissions NOT granted")
                print("\n" + "=" * 70)
                print("PERMISSION SETUP REQUIRED")
                print("=" * 70)
                print("\nTo grant notification permissions:")
                print("1. Open System Settings (System Preferences)")
                print("2. Go to Notifications & Focus")
                print("3. Find 'GUM Notifier' or 'Terminal' in the list")
                print("4. Enable 'Allow Notifications'")
                print("5. Make sure 'Banners' or 'Alerts' is selected")
                print("\nAlternatively, run this test again and grant permission when prompted.")
                return False
            else:
                print(f"⚠️  WARNING: Unknown error: {result.stderr}")
                return False
        elif result.returncode == 0:
            output = result.stdout.strip().lower()
            if output in ["thanks", "not_now", "no_response"]:
                print(f"✅ PASS: Permissions granted! Notification displayed.")
                print(f"   User response: {output}")
                print("\n" + "=" * 70)
                print("SUCCESS: Notifications are working!")
                print("=" * 70)
                return True
            else:
                print(f"⚠️  WARNING: Unexpected output: {output}")
                return False
        else:
            print(f"⚠️  WARNING: Unexpected exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  TIMEOUT: No response within 30 seconds")
        print("   This could mean:")
        print("   - Permissions dialog is waiting for your response")
        print("   - Notification is displayed and waiting for button click")
        print("\n   Check your screen for:")
        print("   - System Settings permission dialog")
        print("   - Notification banner with buttons")
        return None  # Indeterminate
    except Exception as e:
        print(f"❌ FAIL: Error testing permissions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_notification():
    """Test 4: Display a test notification with buttons."""
    print("\n" + "=" * 70)
    print("TEST 4: Display Test Notification")
    print("=" * 70)
    
    swift_path = get_swift_binary_path()
    
    print("Displaying test notification...")
    print("Look for a notification banner with 'Thanks!' and 'Not now!' buttons")
    print("Click one of the buttons to complete the test.")
    print()
    
    try:
        result = subprocess.run(
            [str(swift_path), "🧪 Test Notification", 
             "This is a test notification. Please click 'Thanks!' or 'Not now!' to verify buttons work."],
            capture_output=True,
            text=True,
            timeout=35
        )
        
        if result.returncode == 0:
            feedback = result.stdout.strip().lower()
            if feedback in ["thanks", "not_now"]:
                print(f"✅ SUCCESS: Button clicked! Response: {feedback}")
                return True
            elif feedback == "no_response":
                print("⚠️  No button clicked (timeout after 30 seconds)")
                print("   This is okay - the notification was displayed")
                return True
            else:
                print(f"⚠️  Unexpected response: {feedback}")
                return False
        else:
            print(f"❌ FAIL: Exit code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  TIMEOUT: Test took longer than 35 seconds")
        print("   Check if notification appeared and try clicking a button")
        return None
    except Exception as e:
        print(f"❌ FAIL: Error: {e}")
        return False

def main():
    """Run all permission tests."""
    print("\n" + "=" * 70)
    print("Swift Notification Permission Test Suite")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test 1: Binary exists
    results['binary_exists'] = test_binary_exists()
    if not results['binary_exists']:
        print("\n❌ Cannot continue - binary not found. Build it first.")
        return 1
    
    # Test 2: Binary usage
    results['binary_usage'] = test_binary_usage()
    
    # Test 3: Permissions
    print("\n" + "⚠️  " * 20)
    print("IMPORTANT: The next test will request notification permissions.")
    print("You may see a macOS permission dialog - please grant permission.")
    print("⚠️  " * 20)
    time.sleep(2)
    
    results['permissions'] = test_notification_permissions()
    
    # Test 4: Display notification (only if permissions work)
    if results['permissions']:
        print("\n" + "⚠️  " * 20)
        print("The next test will display a notification with buttons.")
        print("Please click one of the buttons when it appears.")
        print("⚠️  " * 20)
        time.sleep(2)
        
        results['display'] = test_display_notification()
    else:
        print("\n⏭️  Skipping display test - permissions not granted")
        results['display'] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Binary exists:        {'✅' if results['binary_exists'] else '❌'}")
    print(f"Binary usage:          {'✅' if results['binary_usage'] else '❌'}")
    print(f"Permissions granted:   {'✅' if results['permissions'] else '❌' if results['permissions'] is False else '⚠️  (indeterminate)'}")
    print(f"Display test:          {'✅' if results['display'] else '❌' if results['display'] is False else '⏭️  (skipped)' if results['display'] is None else '⚠️  (indeterminate)'}")
    print("=" * 70)
    
    if all(r for r in results.values() if r is not None):
        print("\n✅ ALL TESTS PASSED - Notification system is ready!")
        return 0
    elif results.get('permissions') is False:
        print("\n❌ PERMISSIONS REQUIRED - Please grant notification permissions")
        print("   See instructions above, then run this test again.")
        return 1
    else:
        print("\n⚠️  SOME TESTS INCOMPLETE - Review output above")
        return 2

if __name__ == '__main__':
    sys.exit(main())

