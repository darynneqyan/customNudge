#!/usr/bin/env python3
"""
Debug script to test batch processing manually
"""
import asyncio
import logging
from pathlib import Path
from persistqueue import Queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")

async def test_batch_processing():
    """Test if we can manually trigger batch processing"""
    
    # Check queue
    queue_dir = Path.home() / '.cache' / 'gum' / 'batches' / 'queue'
    queue = Queue(path=str(queue_dir))
    
    print(f"Queue size: {queue.qsize()}")
    
    if queue.qsize() >= 3:
        print("Queue has enough items for batch processing")
        
        # Try to pop a batch manually
        try:
            batch = []
            for _ in range(min(10, queue.qsize())):
                item = queue.get_nowait()
                batch.append(item)
                print(f"Popped item: {item.get('observer_name', 'unknown')}")
            
            print(f"Successfully popped {len(batch)} items")
            print(f"Remaining queue size: {queue.qsize()}")
            
            # Check if items are valid
            if batch:
                print(f"First item keys: {list(batch[0].keys())}")
                print(f"First item content preview: {str(batch[0].get('content', ''))[:100]}...")
            
        except Exception as e:
            print(f"Error popping batch: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Queue doesn't have enough items for batch processing")

if __name__ == "__main__":
    asyncio.run(test_batch_processing())
