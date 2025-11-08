#!/usr/bin/env python3
"""
Test with notification-worthy mock data to verify the notification system works.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gum import gum
from gum.observers import Screen
from gum.models import Observation, Proposition
from sqlalchemy import select

async def test_notification_worthy_data():
    """Test with mock data that should trigger notifications."""
    
    print("=== TESTING NOTIFICATION-WORTHY DATA ===\n")
    
    # Set up environment
    os.environ['GOOGLE_API_KEY'] = "AIzaSyBKYsQ77txAwduIif62yQPanVx2QYG7CuE"
    
    # Clear any existing queue
    queue_dir = Path.home() / ".cache" / "gum" / "batches" / "queue"
    if queue_dir.exists():
        import shutil
        shutil.rmtree(queue_dir)
        print(f"Cleared existing queue at {queue_dir}")
    
    # Create GUM instance
    print("1. Creating GUM instance...")
    gum_instance = gum(
        "test_user",
        "gemini-2.5-flash",
        Screen("gemini-2.5-flash"),
        min_batch_size=3,
        max_batch_size=10,
        enable_notifications=True
    )
    
    print("2. Connecting to database...")
    await gum_instance.connect_db()
    
    # Create notification-worthy mock observations
    print("\n3. Creating notification-worthy mock observations...")
    mock_observations = [
        {
            'observer_name': 'Screen',
            'content': 'User has been browsing social media for 45 minutes straight, switching between Twitter, Instagram, and Facebook repeatedly',
            'content_type': 'text'
        },
        {
            'observer_name': 'Screen', 
            'content': 'User opened YouTube and started watching cat videos instead of working on their important project deadline',
            'content_type': 'text'
        },
        {
            'observer_name': 'Screen',
            'content': 'User has been procrastinating for 2 hours, avoiding their work tasks and getting distracted by notifications',
            'content_type': 'text'
        }
    ]
    
    # Add observations to queue manually
    print("4. Adding observations to queue...")
    observation_ids = []
    for i, obs_data in enumerate(mock_observations):
        obs_id = gum_instance.batcher.push(
            obs_data['observer_name'],
            obs_data['content'], 
            obs_data['content_type']
        )
        observation_ids.append(obs_id)
        print(f"   Added observation {i+1}: {obs_id}")
    
    # Check queue state
    queue_size = gum_instance.batcher.size()
    print(f"   Queue size: {queue_size}")
    
    # Check if batch should be ready
    should_process = gum_instance.batcher.should_process_batch()
    print(f"   Should process batch: {should_process}")
    
    if should_process:
        print("\n5. Processing batch...")
        
        # Get the batch
        batch = gum_instance.batcher.pop_batch()
        print(f"   Popped batch of {len(batch)} observations")
        
        # Process the batch
        try:
            await gum_instance._process_batch(batch)
            print("   Batch processing completed successfully")
        except Exception as e:
            print(f"   ERROR in batch processing: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
    
    print("\n6. Test completed - check the logs above for notification decisions")

if __name__ == "__main__":
    asyncio.run(test_notification_worthy_data())
