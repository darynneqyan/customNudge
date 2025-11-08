#!/usr/bin/env python3
"""
Debug logger for tracking batch processing and notification decision steps.
Writes step-by-step debugging information to a JSON file for GUI display.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock


class DebugLogger:
    """Logger that writes debug steps to a JSON file for GUI consumption."""
    
    def __init__(self, user_name: str, data_directory: str = "~/.cache/gum"):
        """
        Initialize the debug logger.
        
        Args:
            user_name: Name of the user
            data_directory: Directory for storing debug logs
        """
        self.user_name = user_name
        data_directory = Path(data_directory).expanduser()
        data_directory.mkdir(parents=True, exist_ok=True)
        
        self.debug_file = data_directory / f"debug_log_{user_name.lower().replace(' ', '_')}.json"
        self.logger = logging.getLogger("gum.debug")
        self._lock = Lock()
        
        # Initialize file with empty list if it doesn't exist
        if not self.debug_file.exists():
            self._write_logs([])
    
    def _write_logs(self, logs: List[Dict[str, Any]]):
        """Write logs to file (thread-safe)."""
        with self._lock:
            try:
                with open(self.debug_file, 'w') as f:
                    json.dump(logs, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error writing debug log: {e}")
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Read logs from file (thread-safe)."""
        with self._lock:
            try:
                if not self.debug_file.exists():
                    return []
                with open(self.debug_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading debug log: {e}")
                return []
    
    def _add_log_entry(self, entry: Dict[str, Any]):
        """Add a new log entry to the file."""
        logs = self._read_logs()
        logs.append(entry)
        # Keep only last 1000 entries to prevent file from growing too large
        if len(logs) > 1000:
            logs = logs[-1000:]
        self._write_logs(logs)
    
    def log_batch_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log a batch processing event.
        
        Args:
            event_type: Type of event (e.g., 'observation_added', 'batch_ready', 'batch_processing_started', etc.)
            details: Additional details about the event
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'category': 'batch',
            'event_type': event_type,
            'details': details
        }
        self._add_log_entry(entry)
        self.logger.debug(f"Batch event: {event_type} - {details}")
    
    def log_notification_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log a notification decision event.
        
        Args:
            event_type: Type of event (e.g., 'context_gathering_started', 'similar_propositions_found', 'llm_decision_made', etc.)
            details: Additional details about the event
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'category': 'notification',
            'event_type': event_type,
            'details': details
        }
        self._add_log_entry(entry)
        self.logger.debug(f"Notification event: {event_type} - {details}")
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent log entries."""
        logs = self._read_logs()
        return logs[-limit:]
    
    def clear_logs(self):
        """Clear all debug logs."""
        self._write_logs([])
        self.logger.info("Cleared debug logs")

