#!/usr/bin/env python3
"""
Debug batch processing and notification integration
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from gum.gum import gum
from gum.observers.screen import Screen
from gum.notifier import GUMNotifier
from gum.batcher import ObservationBatcher

async def test_batch_processing():
    """Test the complete batch processing pipeline"""
    print("🔍 Testing Batch Processing Pipeline")
    print("=" * 50)
    
    # Test 1: Check if batcher can be created and started
    print("\n1. Testing ObservationBatcher initialization...")
    try:
        batcher = ObservationBatcher("~/.cache/gum", min_batch_size=3, max_batch_size=10)
        await batcher.start()
        print("✅ Batcher initialized and started successfully")
        
        # Check queue size
        queue_size = batcher._queue.qsize()
        print(f"📊 Current queue size: {queue_size}")
        
    except Exception as e:
        print(f"❌ Batcher initialization failed: {e}")
        return
    
    # Test 2: Test notification decision making
    print("\n2. Testing notification decision making...")
    try:
        # Create a mock GUM instance for testing
        class MockGUM:
            def __init__(self):
                from gum.providers import create_provider
                self.provider = create_provider(
                    model="gemini-2.5-flash",
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
        
        mock_gum = MockGUM()
        notifier = GUMNotifier("Eyrin", gum_instance=mock_gum)
        
        # Test notification decision
        decision = await notifier._make_notification_decision(
            observation_content="Test observation: User is working on coding project",
            generated_propositions=[{'text': 'User is actively coding', 'confidence': 8}],
            similar_propositions=[],
            similar_observations=[]
        )
        
        if decision:
            print(f"✅ Notification decision made: should_notify={decision.should_notify}")
            print(f"   Reasoning: {decision.reasoning}")
        else:
            print("❌ No notification decision made - LLM returned None")
            
    except Exception as e:
        print(f"❌ Notification decision test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test batch processing with real GUM instance
    print("\n3. Testing batch processing with real GUM...")
    try:
        async with gum(
            "Eyrin",
            "gemini-2.5-flash",
            Screen("gemini-2.5-flash"),
            min_batch_size=3,
            max_batch_size=10,
            enable_notifications=True
        ) as gum_instance:
            print("✅ GUM instance created successfully")
            
            # Check if notifier is properly initialized
            if gum_instance.notifier:
                print("✅ Notifier is initialized")
                print(f"   Notifier gum_instance: {gum_instance.notifier.gum_instance is not None}")
                print(f"   Notifier provider: {gum_instance.notifier.gum_instance.provider is not None}")
            else:
                print("❌ Notifier is not initialized")
            
            # Wait a bit for some observations to be collected
            print("⏳ Waiting for observations to be collected...")
            await asyncio.sleep(10)
            
            # Check queue status
            if gum_instance.batcher:
                queue_size = gum_instance.batcher._queue.qsize()
                print(f"📊 Queue size after 10 seconds: {queue_size}")
                
                if queue_size >= 3:
                    print("✅ Queue has enough items for batch processing")
                else:
                    print("⚠️  Queue doesn't have enough items yet")
            
    except Exception as e:
        print(f"❌ GUM instance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check notification decisions file
    print("\n4. Checking notification decisions...")
    decisions_file = Path("notification_decisions_eyrin.json")
    if decisions_file.exists():
        with open(decisions_file, 'r') as f:
            decisions = json.load(f)
        
        print(f"📊 Total decisions: {len(decisions)}")
        print(f"📊 Decisions to notify: {sum(1 for d in decisions if d.get('should_notify', False))}")
        print(f"📊 Decisions not to notify: {sum(1 for d in decisions if not d.get('should_notify', False))}")
        
        if decisions:
            latest = decisions[-1]
            print(f"📊 Latest decision timestamp: {latest.get('timestamp', 'N/A')}")
            print(f"📊 Latest should_notify: {latest.get('should_notify', 'N/A')}")
    else:
        print("❌ No notification decisions file found")
    
    # Test 5: Check database activity
    print("\n5. Checking database activity...")
    try:
        import sqlite3
        db_path = Path.home() / '.cache/gum/gum.db'
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check recent observations
            cursor.execute('SELECT COUNT(*) FROM observations')
            total_obs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM propositions')
            total_props = cursor.fetchone()[0]
            
            print(f"📊 Total observations: {total_obs}")
            print(f"📊 Total propositions: {total_props}")
            
            conn.close()
        else:
            print("❌ Database not found")
    except Exception as e:
        print(f"❌ Database check failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_batch_processing())
