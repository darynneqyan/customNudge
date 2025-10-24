#!/usr/bin/env python3
"""
Test script for Adaptive Nudge Engine

This script tests the core functionality of the adaptive nudge engine
without requiring the full GUM system to be running.
"""

import asyncio
import json
import logging
from datetime import datetime
from gum.adaptive_nudge import SystemStateCapture, ObservationWindowManager, LLMJudge, TrainingDataLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_system_state_capture():
    """Test the system state capture functionality."""
    print("Testing System State Capture...")
    
    capture = SystemStateCapture()
    state = capture.capture_system_state()
    
    print(f"Captured state: {json.dumps(state, indent=2)}")
    return state

async def test_llm_judge():
    """Test the LLM judge functionality."""
    print("Testing LLM Judge...")
    
    judge = LLMJudge()
    
    # Create mock nudge and screenshot
    nudge = "Take a 5-minute break from coding"
    screenshot = {
        "active_app": "Visual Studio Code",
        "window_title": "test.py - Visual Studio Code",
        "browser_tabs": [{"browser": "Safari", "url": "https://github.com", "index": 0}],
        "open_files": [{"editor": "VS Code", "path": "/Users/test/project/test.py"}],
        "recent_apps": ["Visual Studio Code", "Terminal", "Safari"],
        "system_info": {"user": "test", "hostname": "test-mac"},
        "clipboard_history": ["def test_function():"]
    }
    
    try:
        result = await judge.get_judge_score(nudge, screenshot)
        print(f"Judge result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"LLM Judge test failed (expected if no API key): {e}")
        return {"score": 0, "reasoning": "Test mode - no API key"}

async def test_observation_window():
    """Test the observation window management."""
    print("Testing Observation Window Manager...")
    
    manager = ObservationWindowManager("test_user")
    
    # Create mock nudge data
    nudge_data = {
        "user_context": {
            "observation_content": "User is coding in VS Code",
            "generated_propositions": [{"text": "User is focused on coding", "confidence": 8}],
            "similar_propositions": [],
            "similar_observations": []
        },
        "nudge_content": "Take a 5-minute break from coding",
        "nudge_type": "break",
        "observation_duration": 10  # 10 seconds for testing
    }
    
    # Start observation
    nudge_id = await manager.start_observation(nudge_data)
    print(f"Started observation: {nudge_id}")
    
    # Check active observations
    active = manager.get_active_observations()
    print(f"Active observations: {len(active)}")
    
    # Wait for completion
    print("Waiting for observation to complete...")
    await asyncio.sleep(15)  # Wait longer than observation duration
    
    # Check statistics
    stats = manager.get_statistics()
    print(f"Observation statistics: {json.dumps(stats, indent=2)}")

async def test_training_logger():
    """Test the training data logger."""
    print("Testing Training Data Logger...")
    
    logger = TrainingDataLogger("test_user")
    
    # Create mock training entry
    training_entry = {
        "nudge_id": "test_nudge_001",
        "timestamp": datetime.now().isoformat(),
        "user_context": {"activity": "coding"},
        "nudge_content": "Take a break",
        "post_nudge_screenshot": {"active_app": "Safari"},
        "judge_score": 1,
        "judge_reasoning": "User switched to browser, indicating break",
        "observation_duration": 180
    }
    
    # Log training data
    await logger.log_training_data(training_entry)
    print("Logged training data")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"Training data statistics: {json.dumps(stats, indent=2)}")

async def main():
    """Run all tests."""
    print("Adaptive Nudge Engine Test Suite")
    print("=" * 50)
    
    try:
        # Test system state capture
        await test_system_state_capture()
        print()
        
        # Test LLM judge
        await test_llm_judge()
        print()
        
        # Test training logger
        await test_training_logger()
        print()
        
        # Test observation window (this will take some time)
        await test_observation_window()
        print()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
