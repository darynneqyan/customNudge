#!/usr/bin/env python3
"""
Test batch processing functionality
"""
import asyncio
import sys
import os
sys.path.append('/Users/eyrinkim/Documents/GitHub/customNudge')

from gum.batcher import ObservationBatcher

async def test_batcher():
    """Test the batcher functionality"""
    batcher = ObservationBatcher("~/.cache/gum", min_batch_size=3, max_batch_size=10)
    
    print(f"Initial queue size: {batcher.size()}")
    print(f"Should process batch: {batcher.should_process_batch()}")
    
    # Start the batcher
    await batcher.start()
    
    print(f"After start - queue size: {batcher.size()}")
    print(f"Should process batch: {batcher.should_process_batch()}")
    
    # Test if we can wait for batch ready
    print("Testing wait_for_batch_ready...")
    try:
        # Wait for 5 seconds to see if the event triggers
        await asyncio.wait_for(batcher.wait_for_batch_ready(), timeout=5.0)
        print("✅ Batch ready event triggered!")
    except asyncio.TimeoutError:
        print("❌ Batch ready event did NOT trigger within 5 seconds")
        print(f"Queue size: {batcher.size()}")
        print(f"Should process: {batcher.should_process_batch()}")
    
    await batcher.stop()

if __name__ == "__main__":
    asyncio.run(test_batcher())
