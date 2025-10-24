#!/usr/bin/env python3
"""
Training Data Logger for Adaptive Nudge Engine

Logs structured training data for policy model development in JSONL format.
This data will be used to train more sophisticated nudge policies.

Goal: Create a comprehensive dataset of nudge effectiveness that can be
used to develop machine learning models for better nudge timing and content.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class TrainingDataLogger:
    """
    Logs training data for nudge policy model development.
    
    This class handles the structured logging of nudge effectiveness data
    in JSONL format, making it easy to process for machine learning.
    """
    
    def __init__(self, user_name: str, data_directory: str = "~/.cache/gum"):
        """
        Initialize the training data logger.
        
        Args:
            user_name: Name of the user being monitored
            data_directory: Base directory for storing training data
        """
        self.user_name = user_name
        self.data_directory = Path(data_directory).expanduser()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Create training data file path
        self.training_file = self.data_directory / f"training_data_{user_name.lower().replace(' ', '_')}.jsonl"
        self.logger = logging.getLogger("gum.adaptive_nudge.training_logger")
        
        self.logger.info(f"TrainingDataLogger initialized for user: {user_name}")
        self.logger.info(f"Training data file: {self.training_file}")
    
    async def log_training_data(self, training_entry: Dict[str, Any]) -> None:
        """
        Log a training data entry to JSONL file.
        
        This method handles the structured logging of nudge effectiveness
        data, ensuring all data is JSON serializable and properly formatted.
        
        Args:
            training_entry: Dictionary containing training data with keys:
                - nudge_id: Unique identifier for the nudge
                - timestamp: When the nudge was sent
                - user_context: Context about user state when nudge was sent
                - nudge_content: The actual nudge message
                - post_nudge_screenshot: System state after observation period
                - judge_score: LLM evaluation (0 or 1)
                - judge_reasoning: LLM's reasoning for the score
                - observation_duration: How long the observation lasted
        """
        try:
            # Ensure all data is JSON serializable
            serializable_entry = self._make_serializable(training_entry)
            
            # Add metadata
            serializable_entry["logged_at"] = datetime.now().isoformat()
            serializable_entry["user_name"] = self.user_name
            
            # Append to JSONL file
            with open(self.training_file, 'a') as f:
                json.dump(serializable_entry, f)
                f.write('\n')
            
            self.logger.info(f"Logged training data for nudge {training_entry.get('nudge_id', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error logging training data: {e}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Recursively converts objects to JSON-serializable types,
        handling datetime objects and other non-serializable types.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def get_training_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get training data entries from the JSONL file.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            List of training data entries
        """
        try:
            entries = []
            with open(self.training_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
                        if limit and len(entries) >= limit:
                            break
            return entries
        except FileNotFoundError:
            self.logger.warning("Training data file not found")
            return []
        except Exception as e:
            self.logger.error(f"Error reading training data: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the training data.
        
        Returns:
            Dict with training data statistics including:
            - total_entries: Number of training entries
            - effective_nudges: Number of effective nudges (score=1)
            - ineffective_nudges: Number of ineffective nudges (score=0)
            - effectiveness_rate: Percentage of effective nudges
            - latest_entry: Most recent training entry
            - nudge_types: Distribution of nudge types
        """
        entries = self.get_training_data()
        if not entries:
            return {"total_entries": 0}
        
        # Calculate effectiveness statistics
        effective_count = sum(1 for entry in entries if entry.get('judge_score') == 1)
        ineffective_count = len(entries) - effective_count
        effectiveness_rate = effective_count / len(entries) if entries else 0
        
        # Analyze nudge types
        nudge_types = {}
        for entry in entries:
            nudge_type = entry.get('nudge_type', 'unknown')
            nudge_types[nudge_type] = nudge_types.get(nudge_type, 0) + 1
        
        return {
            "total_entries": len(entries),
            "effective_nudges": effective_count,
            "ineffective_nudges": ineffective_count,
            "effectiveness_rate": effectiveness_rate,
            "latest_entry": entries[-1] if entries else None,
            "nudge_types": nudge_types,
            "file_path": str(self.training_file)
        }
    
    def export_training_data(self, output_file: str, format: str = "jsonl") -> None:
        """
        Export training data to a file in specified format.
        
        Args:
            output_file: Path to output file
            format: Export format ("jsonl" or "json")
        """
        try:
            entries = self.get_training_data()
            
            if format == "jsonl":
                with open(output_file, 'w') as f:
                    for entry in entries:
                        json.dump(entry, f)
                        f.write('\n')
            elif format == "json":
                with open(output_file, 'w') as f:
                    json.dump(entries, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(entries)} entries to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            raise
    
    def clear_training_data(self) -> None:
        """
        Clear all training data.
        
        WARNING: This permanently deletes all training data.
        """
        try:
            if self.training_file.exists():
                self.training_file.unlink()
                self.logger.info("Training data cleared")
            else:
                self.logger.info("No training data to clear")
        except Exception as e:
            self.logger.error(f"Error clearing training data: {e}")
            raise
    
    def get_recent_entries(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get training entries from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent training entries
        """
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_entries = []
            
            with open(self.training_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry.get('timestamp', '1970-01-01T00:00:00')).timestamp()
                        if entry_time >= cutoff_time:
                            recent_entries.append(entry)
            
            return recent_entries
            
        except FileNotFoundError:
            return []
        except Exception as e:
            self.logger.error(f"Error getting recent entries: {e}")
            return []
