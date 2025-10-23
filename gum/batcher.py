import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
from persistqueue import Queue

class ObservationBatcher:
    """A persistent queue for batching observations to reduce API calls."""
    
    def __init__(self, data_directory: str, min_batch_size: int = 5, max_batch_size: int = 50):
        self.data_directory = Path(data_directory)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Create persistent queue backed by SQLite
        queue_dir = self.data_directory / "batches"
        queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue = Queue(path=str(queue_dir / "queue"))
        
        self._batch_ready_event = asyncio.Event()        
        self.logger = logging.getLogger("gum.batcher")
        
    async def start(self):
        """Start the batching system."""
        self.logger.info(f"Started batcher with {self._queue.qsize()} items in queue")
        
        if self.should_process_batch():
            self._batch_ready_event.set()
        
    async def stop(self):
        """Stop the batching system."""
        self.logger.info("Stopped batcher")
        
    def push(self, observer_name: str, content: str, content_type: str) -> str:
        """Push an observation onto the queue.
        
        Args:
            observer_name: Name of the observer
            content: Observation content
            content_type: Type of content
            
        Returns:
            str: Observation ID
        """
        observation_id = str(uuid.uuid4())
        observation_dict = {
            'id': observation_id,
            'observer_name': observer_name,
            'content': content,
            'content_type': content_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add to queue - automatically persisted by persist-queue
        self._queue.put(observation_dict)
        self.logger.debug(f"Pushed observation {observation_id} to queue (size: {self._queue.qsize()})")
        
        # Signal that a batch is ready if we've reached minimum size
        if self.should_process_batch():
            self._batch_ready_event.set()
        
        return observation_id
        
    def size(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()
        
    def should_process_batch(self) -> bool:
        """Check if the batch should be processed based on minimum batch size."""
        return self._queue.qsize() >= self.min_batch_size
        
    def pop_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pop a batch of observations from the front of the queue (FIFO).
        
        Args:
            batch_size: Number of items to pop. Defaults to max_batch_size
            
        Returns:
            List of observation dictionaries popped from queue
        """
        batch_size = batch_size or self.max_batch_size
        
        batch = []
        for _ in range(min(batch_size, self._queue.qsize())):
            batch.append(self._queue.get_nowait())
        
        if batch:
            self.logger.debug(f"Popped batch of {len(batch)} observations (queue size: {self._queue.qsize()})")
        
        if not self.should_process_batch():
            self._batch_ready_event.clear()
        
        return batch
    
    async def wait_for_batch_ready(self):
        """Wait for a batch to be ready for processing."""
        await self._batch_ready_event.wait()