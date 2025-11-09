#!/usr/bin/env python3
"""
Debug script to check queue status and manually trigger batch processing
"""
import asyncio
import json
from pathlib import Path
from persistqueue import Queue

async def debug_queue():
    queue_dir = Path.home() / '.cache' / 'gum' / 'batches' / 'queue'
    queue = Queue(path=str(queue_dir))
    
    print(f"Queue size: {queue.qsize()}")
    print(f"Queue empty: {queue.empty()}")
    
    # Try to process a few items manually
    if queue.qsize() > 0:
        print("\nTrying to process first few items...")
        try:
            for i in range(min(3, queue.qsize())):
                item = queue.get_nowait()
                print(f"Item {i+1}: {type(item)}")
                if isinstance(item, dict):
                    print(f"  Keys: {list(item.keys())}")
                    print(f"  Observer: {item.get('observer_name', 'N/A')}")
                    print(f"  Content preview: {str(item.get('content', ''))[:50]}...")
                else:
                    print(f"  Value: {str(item)[:100]}...")
        except Exception as e:
            print(f"Error processing items: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_queue())
