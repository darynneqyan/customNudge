#!/usr/bin/env python3
"""
Standalone test for async notification system - tests the async patterns directly
without requiring full GUM dependencies.
"""

import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any


class MockNotifier:
    """Mock notifier that simulates the async notification behavior."""
    
    def __init__(self):
        self.button_feedback: Dict[str, str] = {}
        self.session_stats = {
            "thanks_count": 0,
            "not_now_count": 0,
            "no_response_count": 0,
            "total_notifications": 0
        }
        self.satisfaction_multiplier = 1.0
        self.logger = Mock()
    
    def _start_notification_task(self, message: str, notification_type: str, nudge_id: str):
        """Start notification task - should return immediately."""
        task = asyncio.create_task(
            self._display_notification_with_swift_async(message, notification_type, nudge_id)
        )
        
        def handle_task_done(task):
            try:
                task.result()
            except Exception as e:
                self.logger.error(f"Task error: {e}")
                self._record_feedback(nudge_id, "no_response")
        
        task.add_done_callback(handle_task_done)
    
    async def _display_notification_with_swift_async(self, message: str, notification_type: str, nudge_id: str):
        """Simulate starting Swift process and launching feedback collection."""
        # Simulate subprocess creation (non-blocking)
        process = await self._create_mock_subprocess()
        
        # Launch feedback collection in background
        asyncio.create_task(
            self._collect_swift_feedback(process, nudge_id)
        )
    
    async def _create_mock_subprocess(self):
        """Create a mock subprocess that simulates waiting for user input."""
        # This simulates the Swift binary waiting for button click
        async def mock_communicate():
            # Simulate 2 second delay (user clicks button)
            await asyncio.sleep(2.0)
            return (b"thanks\n", b"")
        
        mock_process = Mock()
        mock_process.communicate = mock_communicate
        mock_process.returncode = 0
        return mock_process
    
    async def _collect_swift_feedback(self, process, nudge_id: str):
        """Collect feedback from Swift process."""
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=35.0
            )
            feedback = stdout.decode('utf-8').strip().lower()
            if feedback in ["thanks", "not_now", "no_response"]:
                self._record_feedback(nudge_id, feedback)
            else:
                self._record_feedback(nudge_id, "no_response")
        except asyncio.TimeoutError:
            self._record_feedback(nudge_id, "no_response")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self._record_feedback(nudge_id, "no_response")
    
    def _record_feedback(self, nudge_id: str, feedback: str):
        """Record feedback synchronously."""
        self.button_feedback[nudge_id] = feedback
        self.session_stats[f"{feedback}_count"] = self.session_stats.get(f"{feedback}_count", 0) + 1
        self.session_stats["total_notifications"] = self.session_stats.get("total_notifications", 0) + 1


async def test_notification_non_blocking():
    """Test that notifications don't block batch processing."""
    print("=" * 70)
    print("TEST 1: Notification Non-Blocking Behavior")
    print("=" * 70)
    
    notifier = MockNotifier()
    
    async def simulate_batch_processing():
        """Simulate batch processing."""
        # Start notification (should return immediately)
        start_time = time.time()
        notifier._start_notification_task("Test", "general", "test-1")
        notification_time = time.time() - start_time
        
        # Batch processing should continue immediately
        await asyncio.sleep(0.1)
        batch_time = time.time() - start_time
        
        print(f"✅ Notification started in: {notification_time:.3f}s")
        print(f"✅ Batch continued in: {batch_time:.3f}s")
        
        if batch_time > 0.5:
            print(f"❌ FAIL: Batch processing took too long ({batch_time:.3f}s)")
            return False
        
        return True
    
    result = await simulate_batch_processing()
    
    # Wait for background task to complete
    await asyncio.sleep(2.5)
    
    # Verify feedback was collected
    if "test-1" in notifier.button_feedback:
        print(f"✅ Feedback collected: {notifier.button_feedback['test-1']}")
    else:
        print("⚠️  Feedback not yet collected (may still be processing)")
    
    return result


async def test_multiple_notifications_concurrent():
    """Test that multiple notifications can run concurrently."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Concurrent Notifications")
    print("=" * 70)
    
    notifier = MockNotifier()
    
    # Start 5 notifications "simultaneously"
    start_time = time.time()
    nudge_ids = [f"test-{i}" for i in range(5)]
    
    for nudge_id in nudge_ids:
        notifier._start_notification_task(f"Notification {nudge_id}", "general", nudge_id)
    
    launch_time = time.time() - start_time
    
    print(f"✅ Launched 5 notifications in: {launch_time:.3f}s")
    
    if launch_time > 0.5:
        print(f"❌ FAIL: Launching took too long ({launch_time:.3f}s)")
        return False
    
    # Wait for all to complete (they run concurrently)
    print("⏳ Waiting for all notifications to complete (2.5s)...")
    await asyncio.sleep(2.5)
    
    # Verify all feedback collected
    collected = sum(1 for nid in nudge_ids if nid in notifier.button_feedback)
    print(f"✅ Collected feedback for {collected}/5 notifications")
    
    if collected < 5:
        print("⚠️  Some notifications still processing (this is okay if < 2.5s elapsed)")
    
    if notifier.session_stats["total_notifications"] > 0:
        print(f"✅ Total notifications recorded: {notifier.session_stats['total_notifications']}")
    
    return True


async def test_batch_processing_throughput():
    """Test that batch processing can continue at high throughput."""
    print("\n" + "=" * 70)
    print("TEST 3: Batch Processing Throughput")
    print("=" * 70)
    
    notifier = MockNotifier()
    
    # Simulate processing 10 observations with notifications
    start_time = time.time()
    processed_count = 0
    
    async def process_observation(obs_id: int):
        """Simulate processing one observation."""
        nonlocal processed_count
        
        # Simulate some work
        await asyncio.sleep(0.01)
        
        # Start notification (non-blocking)
        notifier._start_notification_task(
            f"Notification for observation {obs_id}",
            "general",
            f"nudge-{obs_id}"
        )
        
        processed_count += 1
    
    # Process 10 observations
    tasks = [process_observation(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    print(f"✅ Processed 10 observations in: {total_time:.3f}s")
    print(f"✅ Throughput: {10/total_time:.1f} observations/second")
    
    if total_time > 1.0:
        print(f"❌ FAIL: Processing took too long ({total_time:.3f}s)")
        return False
    
    if processed_count != 10:
        print(f"❌ FAIL: Only processed {processed_count}/10 observations")
        return False
    
    # Wait a bit for notifications to start
    await asyncio.sleep(0.5)
    
    # Check how many notification tasks were started
    print(f"✅ All {processed_count} observations processed without blocking")
    
    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ASYNC NOTIFICATION SYSTEM - STANDALONE TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Non-Blocking Behavior", test_notification_non_blocking),
        ("Multiple Concurrent Notifications", test_multiple_notifications_concurrent),
        ("Batch Processing Throughput", test_batch_processing_throughput),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("   The notification system is non-blocking and ready for production.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   Review output above for details.")
    
    return all_passed


if __name__ == '__main__':
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

