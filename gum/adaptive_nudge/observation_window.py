#!/usr/bin/env python3
"""
Observation Window Management for Adaptive Nudge Engine

Manages the 2-minute observation period after nudge delivery to capture
user behavior and evaluate nudge effectiveness.

Goal: Implement asynchronous observation windows that don't block the main
GUM system while monitoring user behavior after nudges.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from sqlalchemy import select

from .state_capture import SystemStateCapture
from .llm_judge import LLMJudge
from .training_logger import TrainingDataLogger

@dataclass
class NudgeObservation:
    """
    Represents a nudge and its observation window.
    
    Contains all necessary information to track a nudge through its
    observation period and evaluate its effectiveness.
    """
    nudge_id: str
    timestamp: datetime
    user_context: Dict[str, Any]
    nudge_content: str
    observation_duration: int = 120  # 2 minutes 
    callback: Optional[Callable] = None

class ObservationWindowManager:
    """
    Manages observation windows for nudges.
    
    This class handles the asynchronous observation of user behavior
    after nudge delivery, coordinating state capture and LLM evaluation.
    """
    
    def __init__(self, user_name: str, gum_instance=None, notifier=None):
        """
        Initialize the observation window manager.
        
        Args:
            user_name: Name of the user being monitored
            gum_instance: Reference to GUM instance for database access
            notifier: Optional reference to notifier for updating decision entries
        """
        self.user_name = user_name
        self.gum_instance = gum_instance
        self.notifier = notifier
        self.logger = logging.getLogger("gum.adaptive_nudge.observation_window")
        
        # Initialize components
        self.state_capture = SystemStateCapture()
        self.llm_judge = LLMJudge()
        self.training_logger = TrainingDataLogger(user_name)
        
        # Track active observations
        self.active_observations: Dict[str, NudgeObservation] = {}
        self._observation_tasks: Dict[str, asyncio.Task] = {} # added this line
        
        self.logger.info(f"ObservationWindowManager initialized for user: {user_name}")
    
    async def start_observation(self, nudge_data: Dict[str, Any]) -> str:
        """
        Start observation window for a nudge.
        
        This is the main entry point for starting an observation period.
        It creates a unique observation ID and schedules the observation
        completion after the specified duration.
        
        Args:
            nudge_data: Dictionary containing nudge information with keys:
                - user_context: Context about the user's state when nudge was sent
                - nudge_content: The actual nudge message
                - nudge_type: Type of nudge (focus, break, habit, etc.)
                - observation_duration: Duration in seconds (defaults to 120)
        
        Returns:
            str: Unique observation ID for tracking
        """
        try:
            # Use provided nudge_id if available, otherwise generate one
            nudge_id = nudge_data.get('nudge_id')
            if not nudge_id:
                nudge_id = f"nudge_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(nudge_data)}"
            
            # Create observation object with timezone-aware timestamp
            observation = NudgeObservation(
                nudge_id=nudge_id,
                timestamp=datetime.now(timezone.utc),  # Always use UTC for consistency
                user_context=nudge_data.get('user_context', {}),
                nudge_content=nudge_data.get('nudge_content', ''),
                observation_duration=nudge_data.get('observation_duration', 120)  # 2 minutes default
            )
            
            # Store active observation
            self.active_observations[nudge_id] = observation
            
            # Schedule observation completion and store task
            task = asyncio.create_task(self._complete_observation_after_delay(nudge_id))
            self._observation_tasks[nudge_id] = task  # Track the task
            
            # Add done callback to clean up task tracking (prevents memory leak)
            def cleanup_task(t):
                """Clean up task tracking when task completes or fails."""
                try:
                    t.result()  # Re-raise any exceptions for logging
                except asyncio.CancelledError:
                    pass  # Expected for cancellation
                except Exception as e:
                    self.logger.error(f"Observation task {nudge_id} failed: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                finally:
                    # Clean up tracking (use pop to avoid KeyError if already removed)
                    self.active_observations.pop(nudge_id, None)
                    self._observation_tasks.pop(nudge_id, None)
            
            task.add_done_callback(cleanup_task)
            
            self.logger.info(f"Started observation window for nudge {nudge_id} (duration: {observation.observation_duration}s)")
            return nudge_id
            
        except Exception as e:
            self.logger.error(f"Error starting observation: {e}")
            raise
    
    async def _complete_observation_after_delay(self, nudge_id: str) -> None:
        """
        Complete observation after the specified delay.
        
        This method waits for the observation duration, capturing system state
        snapshots at regular intervals and querying observations from the database.
        Then it gets LLM judgment and logs training data.
        
        Args:
            nudge_id: ID of the observation to complete
        """
        try:
            # Get observation from active observations
            observation = self.active_observations.get(nudge_id)
            if not observation:
                self.logger.error(f"Observation {nudge_id} not found")
                return
            
            self.logger.info(f"Waiting {observation.observation_duration}s for observation {nudge_id}")
            
            # Capture system state snapshots at regular intervals (every 30 seconds)
            snapshot_interval = 30  #hyperparameter, can tune  
            num_snapshots = observation.observation_duration // snapshot_interval
            post_nudge_system_state_snapshots = []
            start_time = observation.timestamp

            # Ensure timezone-aware (should already be UTC from start_observation, but safety check)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC for consistency
                start_time = start_time.astimezone(timezone.utc)
            
            for i in range(num_snapshots + 1):  
                if i > 0:
                    await asyncio.sleep(snapshot_interval)
                
                snapshot_time = start_time + timedelta(seconds=i * snapshot_interval)
                self.logger.debug(f"Capturing system state snapshot {i+1}/{num_snapshots+1} at {snapshot_time} for nudge {nudge_id}")
                
                # Capture snapshot with error handling (continue on failure)
                try:
                    snapshot = self.state_capture.capture_system_state()
                    snapshot['snapshot_timestamp'] = snapshot_time.isoformat()
                    post_nudge_system_state_snapshots.append(snapshot)
                except Exception as e:
                    self.logger.warning(f"Failed to capture snapshot {i+1}/{num_snapshots+1} for {nudge_id}: {e}")
                    # Continue to next snapshot - don't fail entire observation
                    continue
            
            # Query observations from database during the 2 min window
            observations = []
            if self.gum_instance:
                try:
                    end_time = start_time + timedelta(seconds=observation.observation_duration)
                    # Ensure timezone-aware (should already be UTC, but safety check)
                    if end_time.tzinfo is None:
                        end_time = end_time.replace(tzinfo=timezone.utc)
                    else:
                        end_time = end_time.astimezone(timezone.utc)
                    
                    observations = await self._query_observations_during_window(
                        start_time=start_time,
                        end_time=end_time
                    )
                    self.logger.info(f"Found {len(observations)} observations during window")
                except Exception as e:
                    self.logger.warning(f"Could not query observations: {e}")
            
            # LLM judge on effectiveness (with error handling)
            self.logger.info(f"Getting LLM judgment for {nudge_id}")
            try:
                judge_score = await self.llm_judge.get_judge_score(
                    nudge=observation.nudge_content,
                    observations=observations,
                    post_nudge_system_state_snapshots=post_nudge_system_state_snapshots
                )
            except Exception as e:
                self.logger.error(f"LLM judge failed for {nudge_id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Use default score to at least log the data (partial data is better than no data)
                judge_score = {
                    "score": 0,
                    "reasoning": f"LLM judge failed: {str(e)}",
                    "compliance_percentage": 0,
                    "pattern": "Error"
                }
            
            # Prepare training data entry
            training_entry = {
                "nudge_id": nudge_id,
                "timestamp": observation.timestamp.isoformat(),
                "user_context": observation.user_context,
                "nudge_content": observation.nudge_content,
                "observations": observations,
                "post_nudge_system_state_snapshots": post_nudge_system_state_snapshots,
                "judge_score": judge_score["score"],
                "judge_reasoning": judge_score["reasoning"],
                "observation_duration": observation.observation_duration
            }
            
            # Add in-context learning fields from decision entry if available
            if self.notifier:
                try:
                    # Validate nudge_id before lookup
                    if not nudge_id:
                        self.logger.warning("nudge_id is None, cannot find decision entry")
                    else:
                        # Make a snapshot to avoid concurrent modification errors
                        decisions_snapshot = list(self.notifier.decisions_log)
                        decision_entry = None
                        for decision in decisions_snapshot:
                            if decision.get('nudge_id') == nudge_id:
                                decision_entry = decision
                                break
                        
                        if decision_entry:
                            # Add new fields if they exist (backward compatible)
                            training_entry.update({
                                "policy_version": decision_entry.get('policy_version'),
                                "time_since_last_nudge": decision_entry.get('time_since_last_nudge'),
                                "frequency_context": decision_entry.get('frequency_context'),
                                "examples_available_count": decision_entry.get('examples_available_count'),
                                "examples_used_count": decision_entry.get('examples_used_count', 0)
                            })
                        else:
                            self.logger.debug(f"Decision entry not found for nudge_id: {nudge_id}")
                except Exception as e:
                    self.logger.warning(f"Error adding in-context learning fields to training entry: {e}")
            
            # Log training data (separate error handling - don't fail entire observation)
            try:
                await self.training_logger.log_training_data(training_entry)
            except Exception as e:
                self.logger.error(f"Failed to log training data for {nudge_id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue anyway - we still want to update decision entry
            
            # Update decision entry with effectiveness data if notifier is available (separate error handling)
            if self.notifier:
                try:
                    self.notifier.update_decision_with_effectiveness(nudge_id, judge_score)
                except Exception as e:
                    self.logger.warning(f"Could not update decision entry: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up active observation and task (use pop to avoid KeyError if already removed by callback)
            self.active_observations.pop(nudge_id, None)
            self._observation_tasks.pop(nudge_id, None)
            
            self.logger.info(f"Completed observation for nudge {nudge_id}, score: {judge_score['score']}")
            
        except Exception as e:
            self.logger.error(f"Error completing observation {nudge_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Clean up on error (use pop to avoid KeyError if already removed by callback)
            self.active_observations.pop(nudge_id, None)
            self._observation_tasks.pop(nudge_id, None)
    
    async def _query_observations_during_window(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Query observations from database during the observation window.
        
        Args:
            start_time: Start of observation window (should be timezone-aware)
            end_time: End of observation window (should be timezone-aware)
            
        Returns:
            List of observation dictionaries with content and timestamp
        """
        if not self.gum_instance:
            return []
        
        try:
            from ..models import Observation
            
            # Ensure timezone-aware (convert to UTC for consistent comparison)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                start_time = start_time.astimezone(timezone.utc)
            
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = end_time.astimezone(timezone.utc)
            
            async with self.gum_instance._session() as session:
                # Query observations created during the window
                stmt = select(Observation).where(
                    Observation.created_at >= start_time,
                    Observation.created_at <= end_time
                ).order_by(Observation.created_at)
                
                result = await session.execute(stmt)
                observations = result.scalars().all()
                
                # Convert to dictionaries
                obs_list = []
                for obs in observations:
                    obs_list.append({
                        "id": obs.id,
                        "content": obs.content,
                        "content_type": obs.content_type,
                        "observer_name": obs.observer_name,
                        "created_at": obs.created_at.isoformat() if hasattr(obs.created_at, 'isoformat') else str(obs.created_at),
                        "timestamp": obs.created_at.isoformat() if hasattr(obs.created_at, 'isoformat') else str(obs.created_at)
                    })
                
                return obs_list
                
        except Exception as e:
            self.logger.error(f"Error querying observations: {e}")
            return []
    
    def get_active_observations(self) -> Dict[str, NudgeObservation]:
        """
        Get currently active observations.
        
        Returns:
            Dict mapping nudge_id to NudgeObservation objects
        """
        return self.active_observations.copy()
    
    def cancel_observation(self, nudge_id: str) -> bool:
        """
        Cancel an active observation.
        
        Args:
            nudge_id: ID of the observation to cancel
            
        Returns:
            bool: True if observation was cancelled, False if not found
        """
        if nudge_id in self.active_observations:
            # Cancel task if it exists
            task = self._observation_tasks.pop(nudge_id, None)
            if task:
                task.cancel()
            
            # Remove from active observations (use pop to avoid KeyError if already removed)
            self.active_observations.pop(nudge_id, None)
            return True
        else:
            self.logger.warning(f"Observation {nudge_id} not found for cancellation")
            return False
    
    def get_observation_status(self, nudge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific observation.
        
        Args:
            nudge_id: ID of the observation to check
            
        Returns:
            Dict with observation status, or None if not found
        """
        observation = self.active_observations.get(nudge_id)
        if not observation:
            return None
        
        # Use timezone-aware datetime for calculation (observation.timestamp is already UTC)
        now = datetime.now(timezone.utc)
        elapsed = (now - observation.timestamp).total_seconds()
        remaining = max(0, observation.observation_duration - elapsed)
        
        return {
            "nudge_id": nudge_id,
            "started_at": observation.timestamp.isoformat(),
            "duration": observation.observation_duration,
            "elapsed": elapsed,
            "remaining": remaining,
            "is_active": remaining > 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about active observations.
        
        Returns:
            Dict with observation statistics
        """
        # Snapshot to avoid race condition (dict can change during iteration)
        observations = list(self.active_observations.values())
        
        if not observations:
            return {
                "active_observations": 0,
                "oldest_observation": None,
                "newest_observation": None
            }
        
        timestamps = [obs.timestamp for obs in observations]
        
        return {
            "active_observations": len(observations),
            "oldest_observation": min(timestamps).isoformat(),
            "newest_observation": max(timestamps).isoformat()
        }
