#!/usr/bin/env python3
"""
Trace batch processing flow with mock data to identify where database insertion fails.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gum import gum
from gum.observers import Screen
from gum.batcher import ObservationBatcher
from gum.models import Observation, Proposition
from sqlalchemy import select

async def trace_batch_flow():
    """Trace the complete batch processing flow with mock data."""
    
    print("=== TRACING BATCH PROCESSING FLOW ===\n")
    
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
    
    # Check initial database state
    session = gum_instance.Session()
    obs_result = await session.execute(select(Observation))
    initial_obs_count = len(obs_result.scalars().all())
    prop_result = await session.execute(select(Proposition))
    initial_prop_count = len(prop_result.scalars().all())
    await session.close()
    
    print(f"   Initial observations in DB: {initial_obs_count}")
    print(f"   Initial propositions in DB: {initial_prop_count}")
    
    # Create mock observations manually
    print("\n3. Creating mock observations...")
    mock_observations = [
        {
            'observer_name': 'Screen',
            'content': 'User opened VS Code and started coding Python',
            'content_type': 'text'
        },
        {
            'observer_name': 'Screen', 
            'content': 'User switched to browser and opened YouTube',
            'content_type': 'text'
        },
        {
            'observer_name': 'Screen',
            'content': 'User returned to VS Code and continued coding',
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
        print("\n5. Manually triggering batch processing...")
        
        # Get the batch
        batch = gum_instance.batcher.pop_batch()
        print(f"   Popped batch of {len(batch)} observations")
        
        # Process the batch
        print("6. Processing batch...")
        try:
            await gum_instance._process_batch(batch)
            print("   Batch processing completed successfully")
        except Exception as e:
            print(f"   ERROR in batch processing: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
    
    # Check final database state
    print("\n7. Checking final database state...")
    session = gum_instance.Session()
    obs_result = await session.execute(select(Observation))
    final_obs_count = len(obs_result.scalars().all())
    prop_result = await session.execute(select(Proposition))
    final_prop_count = len(prop_result.scalars().all())
    
    print(f"   Final observations in DB: {final_obs_count}")
    print(f"   Final propositions in DB: {final_prop_count}")
    print(f"   Observations added: {final_obs_count - initial_obs_count}")
    print(f"   Propositions added: {final_prop_count - initial_prop_count}")
    
    # Check what's actually in the database
    if final_obs_count > initial_obs_count:
        recent_obs_result = await session.execute(select(Observation).order_by(Observation.id.desc()).limit(3))
        recent_obs = recent_obs_result.scalars().all()
        print("\n   Recent observations:")
        for obs in recent_obs:
            print(f"     - {obs.content[:50]}...")
    
    if final_prop_count > initial_prop_count:
        recent_props_result = await session.execute(select(Proposition).order_by(Proposition.id.desc()).limit(3))
        recent_props = recent_props_result.scalars().all()
        print("\n   Recent propositions:")
        for prop in recent_props:
            print(f"     - {prop.text[:50]}...")
    
    await session.close()
    
    # Clean up
    await gum_instance.close_db()
    print("\n8. Cleanup completed")

if __name__ == "__main__":
    asyncio.run(trace_batch_flow())
