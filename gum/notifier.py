#!/usr/bin/env python3
"""
Simple GUM Notification Module
Integrates with GUM's batching pipeline and uses BM25 for similarity search.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import asyncio
import subprocess

from .db_utils import search_propositions_bm25, search_observations_bm25, get_related_propositions
from .models import Observation, Proposition
from .prompts.gum import NOTIFICATION_DECISION_PROMPT
from .adaptive_nudge import ObservationWindowManager
import re

# PyObjC imports and delegate class (defined at module level to avoid redefinition errors)
# THIS STUFF DEALS WITH THE NOTIFICATIONS DISPLAYING WITH BUTTONS AS BANNERS
try:
    from Foundation import NSUserNotification, NSUserNotificationCenter, NSObject
    from AppKit import NSApp
    import objc
    
    class NotificationDelegate(NSObject):
        """Delegate class for handling NSUserNotification button clicks."""
        notifier = None
        
        def userNotificationCenter_didActivateNotification_(self, center, notification):
            """Handle notification interaction."""
            user_info = notification.userInfo()
            nudge_id = user_info.get("nudge_id", "unknown") if user_info else "unknown"
            
            # Check which button was clicked
            activation_type = notification.activationType()
            
            if activation_type == 1:  # Action button clicked (Thanks!)
                feedback = "thanks"
            elif activation_type == 2:  # Other button clicked (Not now!)
                feedback = "not_now"
            else:
                feedback = "no_response"
            
            # Store feedback
            if self.notifier:
                self.notifier.button_feedback[nudge_id] = feedback
                self.notifier.session_stats[f"{feedback}_count"] += 1
                self.notifier.logger.info(f"📊 User feedback for {nudge_id}: {feedback}")
                self.notifier._update_satisfaction_multiplier()
    
    PYOJBC_AVAILABLE = True
except ImportError:
    PYOJBC_AVAILABLE = False
    NotificationDelegate = None


@dataclass
class NotificationContext:
    """Context for a notification decision."""
    timestamp: str
    observation_content: str
    observation_id: Optional[int]
    generated_propositions: List[Dict[str, Any]]  # Propositions generated from this observation
    similar_propositions: List[Dict[str, Any]]    # Similar propositions found via BM25
    similar_observations: List[Dict[str, Any]]    # Similar observations found via BM25
    
@dataclass
class NotificationDecision:
    """LLM decision about whether to notify."""
    should_notify: bool
    relevance_score: float
    urgency_score: float
    impact_score: float
    reasoning: str
    notification_message: str
    notification_type: str


class GUMNotifier:
    """Simple notification system that uses BM25 to find similar propositions and observations."""
    
    def __init__(self, user_name: str, gum_instance=None):
        """
        Initialize the notifier.
        
        Args:
            user_name: Name of the user
            gum_instance: Reference to the GUM instance for database access
        """
        self.user_name = user_name
        self.gum_instance = gum_instance
        self.logger = logging.getLogger("gum.notifier")
        
        # Notification log
        self.notification_contexts: List[NotificationContext] = []
        self.sent_notifications: List[Dict[str, Any]] = []
        self.decisions_log: List[Dict[str, Any]] = []
        self.contexts_file = Path(f"notification_contexts_{user_name.lower().replace(' ', '_')}.json")
        self.decisions_file = Path(f"notification_decisions_{user_name.lower().replace(' ', '_')}.json")
        
        # Initialize Adaptive Nudge Engine
        self.observation_manager = ObservationWindowManager(user_name)
        self.adaptive_nudge_enabled = True  # Can be controlled via environment variable
        
        # Load existing contexts if available
        self._load_notification_log()

        # Cooldown configuration
        self.last_notification_time = None
        self.min_notification_interval = 180 # 3 minutes in seconds

        # Session tracking
        self.session_start_time = datetime.now()
        
        # Button feedback tracking (satisfaction score)
        self.button_feedback: Dict[str, str] = {}  # {nudge_id: "thanks" | "not_now" | "no_response"}
        self.session_stats = {
            "thanks_count": 0,
            "not_now_count": 0,
            "no_response_count": 0,
            "total_notifications": 0
        }
        
        # Adaptive thresholds based on satisfaction
        self.satisfaction_multiplier = 1.0  # Starts at normal, increases if user annoyed
    
    def _load_notification_log(self):
        """Load notification contexts from file."""
        if self.contexts_file.exists():
            try:
                with open(self.contexts_file, 'r') as f:
                    data = json.load(f)
                    self.notification_contexts = [NotificationContext(**entry) for entry in data]
                if len(self.notification_contexts) > 0:
                    self.logger.info(f"Loaded {len(self.notification_contexts)} notification contexts")
                else:
                    self.logger.info("No existing notification contexts found - system will create new contexts as observations are processed")
            except Exception as e:
                self.logger.error(f"Error loading notification log: {e}")
                self.notification_contexts = []
        else:
            self.logger.info("Notification contexts file not found - system will create new contexts as observations are processed")
    
    def _save_notification_log(self):
        """Save notification contexts to file."""
        try:
            data = [asdict(ctx) for ctx in self.notification_contexts]
            with open(self.contexts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving notification log: {e}")
    
    def _get_learning_context(self, notification_type: str = None) -> str:
        """Extract learning insights from training data for similar notification types."""
        try:
            from .adaptive_nudge.training_logger import TrainingDataLogger
            training_logger = TrainingDataLogger(self.user_name)
            
            # Get recent training entries
            recent_entries = training_logger.get_recent_entries(hours=24)
            
            if not recent_entries:
                return "No recent training data available for learning."
            
            # Filter by notification type if specified
            if notification_type:
                relevant_entries = [e for e in recent_entries if e.get('nudge_type') == notification_type]
            else:
                relevant_entries = recent_entries
            
            if not relevant_entries:
                return f"No recent {notification_type} notifications to learn from."
            
            # Analyze effectiveness patterns
            effective_count = sum(1 for e in relevant_entries if e.get('judge_score') == 1)
            ineffective_count = len(relevant_entries) - effective_count
            effectiveness_rate = effective_count / len(relevant_entries) if relevant_entries else 0
            
            # Get recent patterns
            recent_patterns = []
            for entry in relevant_entries[-3:]:  # Last 3 entries
                pattern = {
                    'type': entry.get('nudge_type', 'unknown'),
                    'effective': entry.get('judge_score') == 1,
                    'reasoning': entry.get('judge_reasoning', '')[:100] + "..." if entry.get('judge_reasoning') else 'No reasoning',
                    'timestamp': entry.get('timestamp', 'unknown')
                }
                recent_patterns.append(pattern)
            
            # Build learning context
            learning_context = f"""
**Effectiveness Analysis:**
- Recent {notification_type or 'all'} notifications: {len(relevant_entries)}
- Effective: {effective_count} ({effectiveness_rate:.1%})
- Ineffective: {ineffective_count}

**Recent Patterns:**
"""
            for pattern in recent_patterns:
                status = "✅ Effective" if pattern['effective'] else "❌ Ineffective"
                learning_context += f"- [{pattern['timestamp']}] {pattern['type']}: {status}\n  Reasoning: {pattern['reasoning']}\n"
            
            # Add recommendations
            if effectiveness_rate < 0.3:
                learning_context += "\n**Learning Insight:** Low effectiveness rate suggests need for different approach or timing."
            elif effectiveness_rate > 0.7:
                learning_context += "\n**Learning Insight:** High effectiveness rate - continue similar approach."
            else:
                learning_context += "\n**Learning Insight:** Mixed results - consider context-specific adjustments."
            
            return learning_context.strip()
            
        except Exception as e:
            self.logger.error(f"Error getting learning context: {e}")
            return "Unable to retrieve learning context from training data."
    
    def _save_decision(self, context: NotificationContext, decision: NotificationDecision):
        """Save a notification decision to separate file for GUI."""
        try:
            decision_entry = {
                'timestamp': context.timestamp,
                'observation_id': context.observation_id,
                'should_notify': decision.should_notify,
                'relevance_score': decision.relevance_score,
                'urgency_score': decision.urgency_score,
                'impact_score': decision.impact_score,
                'reasoning': decision.reasoning,
                'notification_message': decision.notification_message,
                'notification_type': decision.notification_type,
                'observation_content': context.observation_content[:200] + "...",
                'generated_propositions_count': len(context.generated_propositions),
                'similar_propositions_count': len(context.similar_propositions),
                'similar_observations_count': len(context.similar_observations)
            }
            
            self.decisions_log.append(decision_entry)
            
            self.logger.info(f"Saving decision to {self.decisions_file}")
            with open(self.decisions_file, 'w') as f:
                json.dump(self.decisions_log, f, indent=2)
            self.logger.info(f"Decision saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving decision: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    
    async def _find_similar_propositions(self, propositions: List[Proposition], 
                                        limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar propositions using BM25 search based on generated propositions.
        Uses the same approach as GUM's _generate_and_search method.
        
        Args:
            propositions: List of generated propositions to search with
            limit: Maximum number of similar propositions to return per generated proposition
            
        Returns:
            List of similar propositions with their metadata
        """
        if not self.gum_instance:
            self.logger.warning("No GUM instance available for BM25 search")
            return []
        
        try:
            async with self.gum_instance._session() as session:
                similar_props = []
                seen_prop_ids = set()
                
                # For each generated proposition, find similar ones (like GUM does)
                for prop in propositions:
                    # Skip if this is one we already found
                    if prop.id in seen_prop_ids:
                        continue
                    
                    # Search using proposition text + reasoning (same as GUM's approach)
                    query = f"{prop.text}\n{prop.reasoning}"
                    self.logger.info(f"Searching for propositions similar to: {prop.text[:50]}...")
                    
                    with session.no_autoflush:
                        results = await search_propositions_bm25(
                            session,
                            query,
                            limit=limit,
                            mode="OR",
                            include_observations=False,
                            start_time=datetime.now() - timedelta(days=30),
                            enable_decay=True,
                            enable_mmr=False
                        )
                    
                    # Convert results to dictionaries
                    for similar_prop, score in results:
                        if similar_prop.id not in seen_prop_ids:
                            similar_props.append({
                                "id": similar_prop.id,
                                "text": similar_prop.text,
                                "reasoning": similar_prop.reasoning,
                                "confidence": similar_prop.confidence,
                                "decay": similar_prop.decay,
                                "relevance_score": float(score),
                                "created_at": similar_prop.created_at.isoformat() if hasattr(similar_prop.created_at, 'isoformat') else str(similar_prop.created_at),
                                "matched_with": prop.text[:50] + "..."  # Which generated prop it matched with
                            })
                            seen_prop_ids.add(similar_prop.id)
                
                self.logger.info(f"Found {len(similar_props)} similar propositions")
                return similar_props
                
        except Exception as e:
            self.logger.error(f"Error finding similar propositions: {e}")
            return []
    
    async def _find_similar_observations(self, observation_content: str, 
                                        limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar observations using direct BM25 search on observations.
        
        Args:
            observation_content: The observation content to search for
            limit: Maximum number of similar observations to return
            
        Returns:
            List of similar observations with their metadata
        """
        if not self.gum_instance:
            self.logger.warning("No GUM instance available for observation search")
            return []
        
        try:
            async with self.gum_instance._session() as session:
                self.logger.info(f"Searching for similar observations")
                
                # Search observations directly using BM25
                results = await search_observations_bm25(
                    session,
                    observation_content,  # Use full content as query
                    limit=limit,
                    mode="OR",
                    start_time=datetime.now() - timedelta(days=7),  # Last 7 days
                )
                
                # Convert results to dictionaries
                similar_obs = []
                for obs, score in results:
                    similar_obs.append({
                        "id": obs.id,
                        "observer_name": obs.observer_name,
                        "content": obs.content[:200] + "..." if len(obs.content) > 200 else obs.content,
                        "content_type": obs.content_type,
                        "created_at": obs.created_at.isoformat() if hasattr(obs.created_at, 'isoformat') else str(obs.created_at),
                        "relevance_score": float(score)
                    })
                
                self.logger.info(f"Found {len(similar_obs)} similar observations")
                return similar_obs
                
        except Exception as e:
            self.logger.error(f"Error finding similar observations: {e}")
            return []
    
    async def _make_notification_decision(self, 
                                         observation_content: str,
                                         generated_propositions: List[Dict[str, Any]],
                                         similar_propositions: List[Dict[str, Any]],
                                         similar_observations: List[Dict[str, Any]]) -> Optional[NotificationDecision]:
        """
        Use LLM to decide whether to send a notification.
        
        Args:
            observation_content: The observation text
            generated_propositions: Propositions generated from this observation
            similar_propositions: Similar propositions from history
            similar_observations: Similar observations from history
            
        Returns:
            NotificationDecision or None if error
        """
        # Format generated propositions
        gen_props_text = "\n".join([
            f"- {p['text']} (confidence: {p['confidence']}/10)"
            for p in generated_propositions[:3]
        ]) if generated_propositions else "None"
        
        # Format similar propositions
        sim_props_text = "\n".join([
            f"- {p['text']} (confidence: {p['confidence']}/10, relevance: {p['relevance_score']:.2f})"
            for p in similar_propositions[:5]
        ]) if similar_propositions else "None"
        
        # Format similar observations
        sim_obs_text = "\n".join([
            f"- {o['content'][:100]}... (relevance: {o['relevance_score']:.2f})"
            for o in similar_observations[:5]
        ]) if similar_observations else "None"
        
        # Format notification history (last 5)
        recent_notifs = self.sent_notifications[-5:]
        notif_history_text = "\n".join([
            f"- [{n['timestamp']}] {n['message']} (type: {n['type']})"
            for n in recent_notifs
        ]) if recent_notifs else "No recent notifications"
        
        # Get learning context from training data
        learning_context = self._get_learning_context()
        
        # Construct prompt
        prompt = NOTIFICATION_DECISION_PROMPT.format(
            user_name=self.user_name,
            observation_content=observation_content,
            generated_propositions=gen_props_text,
            similar_propositions=sim_props_text,
            similar_observations=sim_obs_text,
            notification_history=notif_history_text,
            learning_context=learning_context
        )
        
        try:
            # Call LLM using chat_completion (same as GUM does)
            response = await self.gum_instance.provider.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"}
            )
            
            # Debug: Log the response
            self.logger.debug(f"LLM response: {repr(response)}")
            
            if not response or not response.strip():
                self.logger.error("Empty response from LLM")
                return None
            
            # Clean up the response - remove markdown code blocks if present (same as GUM does)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                # Remove ```json from start and ``` from end
                cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
            elif cleaned_response.startswith('```'):
                # Remove ``` from start and end
                cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                self.logger.error(f"Original response: {repr(response)}")
                self.logger.error(f"Cleaned response: {repr(cleaned_response)}")
                return None
            
            # Create decision object
            decision = NotificationDecision(
                should_notify=result.get('should_notify', False),
                relevance_score=float(result.get('relevance_score', 0)),
                urgency_score=float(result.get('urgency_score', 0)),
                impact_score=float(result.get('impact_score', 0)),
                reasoning=result.get('reasoning', ''),
                notification_message=result.get('notification_message', ''),
                notification_type=result.get('notification_type', 'none')
            )
            
            self.logger.info(
                f"LLM Decision: should_notify={decision.should_notify}, "
                f"relevance={decision.relevance_score}, urgency={decision.urgency_score}, "
                f"impact={decision.impact_score}"
            )
            
            return decision
                
        except Exception as e:
            self.logger.error(f"Error making notification decision: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def process_observation_batch(self, batched_observations: List[Dict[str, Any]]):
        """
        Process a batch of observations and find similar propositions/observations.
        
        This method is called after GUM's _process_batch completes.
        By this point, propositions are already attached to observations in the DB.
        
        Args:
            batched_observations: List of observation dictionaries from GUM's batch
        """
        self.logger.info(f"Processing {len(batched_observations)} observations for notifications")
        
        # Process each observation
        for obs_data in batched_observations:
            try:
                # Extract observation data
                observation_content = obs_data.get('content', '')
                observation_id = obs_data.get('id')
                
                self.logger.info(f"Processing observation {observation_id}")
                self.logger.debug(f"Content preview: {observation_content[:100]}...")
                
                # Get propositions that were generated/attached to this observation
                async with self.gum_instance._session() as session:
                    generated_propositions = await get_related_propositions(session, observation_id)
                
                self.logger.info(f"Found {len(generated_propositions)} propositions for observation {observation_id}")
                
                # Find similar propositions using the generated propositions
                similar_propositions = await self._find_similar_propositions(generated_propositions)
                
                # Find similar observations using the observation content
                similar_observations = await self._find_similar_observations(observation_content)
                
                # Convert generated propositions to dict format
                generated_props_dict = []
                for prop in generated_propositions:
                    generated_props_dict.append({
                        "id": prop.id,
                        "text": prop.text,
                        "reasoning": prop.reasoning,
                        "confidence": prop.confidence,
                        "decay": prop.decay,
                        "created_at": prop.created_at.isoformat() if hasattr(prop.created_at, 'isoformat') else str(prop.created_at)
                    })
                
                # Create notification context
                context = NotificationContext(
                    timestamp=datetime.now().isoformat(),
                    observation_content=observation_content,
                    observation_id=observation_id,
                    generated_propositions=generated_props_dict,
                    similar_propositions=similar_propositions,
                    similar_observations=similar_observations
                )
                
                # Store context
                self.notification_contexts.append(context)
                
                # Use LLM to decide whether to notify
                decision = await self._make_notification_decision(
                    observation_content,
                    generated_props_dict,
                    similar_propositions,
                    similar_observations
                )
                
                if decision:
                    self.logger.info(f"LLM made a decision: should_notify={decision.should_notify}")
                    # Save decision to file for GUI
                    self._save_decision(context, decision)
                    
                    # Log decision
                    self.logger.info(
                        f"LLM Decision - Should notify: {decision.should_notify}, "
                        f"Relevance: {decision.relevance_score}/10, "
                        f"Urgency: {decision.urgency_score}/10, "
                        f"Impact: {decision.impact_score}/10"
                    )
                    
                    # Print to console
                    print(f"\n{'='*60}")
                    print(f"LLM NOTIFICATION DECISION")
                    print(f"{'='*60}")
                    print(f"Should Notify: {'✅ YES' if decision.should_notify else '❌ NO'}")
                    print(f"Type: {decision.notification_type}")
                    print(f"Relevance: {decision.relevance_score}/10")
                    print(f"Urgency: {decision.urgency_score}/10")
                    print(f"Impact: {decision.impact_score}/10")
                    print(f"Reasoning: {decision.reasoning}")
                    
                    if decision.should_notify:
                        in_cooldown, remaining = self._is_in_cooldown()
                        if in_cooldown:
                            print(f"NOTIFICATION BLOCKED - Cooldown ({remaining:.0f}s remaining)")
                            print(f"{'='*60}\n")
                            continue

                        print(f"\n📢 NOTIFICATION MESSAGE:")
                        print(f"{decision.notification_message}")
                        print(f"{'='*60}\n")
                        
                        # Display native macOS notification
                        # self._display_notification_with_buttons(decision.notification_message, decision.notification_type, context.observation_id)
                        self._display_notification_with_buttons(
                            decision.notification_message, 
                            decision.notification_type,
                            context.observation_id
                        )
                        
                        # Track sent notification
                        self.sent_notifications.append({
                            'nudge_id': context.observation_id, 
                            'timestamp': context.timestamp,
                            'message': decision.notification_message,
                            'type': decision.notification_type,
                            'relevance': decision.relevance_score,
                            'urgency': decision.urgency_score,
                            'impact': decision.impact_score
                        })
                        
                        # ADAPTIVE NUDGE ENGINE: Start observation window
                        if self.adaptive_nudge_enabled:
                            try:
                                nudge_data = {
                                    'nudge_id': context.observation_id,
                                    'user_context': {
                                        'observation_content': context.observation_content,
                                        'generated_propositions': context.generated_propositions,
                                        'similar_propositions': context.similar_propositions,
                                        'similar_observations': context.similar_observations,
                                        'relevance_score': decision.relevance_score,
                                        'urgency_score': decision.urgency_score,
                                        'impact_score': decision.impact_score
                                    },
                                    'nudge_content': decision.notification_message,
                                    'nudge_type': decision.notification_type,
                                    'observation_duration': 180  # 3 minutes
                                }
                                
                                # Start observation window asynchronously
                                asyncio.create_task(self.observation_manager.start_observation(nudge_data))
                                self.logger.info("Started adaptive nudge observation window")
                                
                            except Exception as e:
                                self.logger.error(f"Error starting adaptive nudge observation: {e}")
                    else:
                        print(f"\n❌ No notification sent")
                        print(f"{'='*60}\n")
                else:
                    self.logger.warning("LLM returned no decision")
                
            except Exception as e:
                self.logger.error(f"Error processing observation for notifications: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Save all contexts to file
        self._save_notification_log()
        self.logger.info(f"Saved notification contexts to {self.contexts_file}")
    
    # def _display_notification(self, message: str, notification_type: str):
    #     """Display a native macOS notification."""
    #     try:
    #         # Escape quotes in the message
    #         escaped_message = message.replace('"', '\\"')
            
    #         # Customize title based on notification type
    #         title_map = {
    #             'break': '⏰ Break Reminder',
    #             'focus': '🎯 Focus Nudge', 
    #             'productivity': '⚡ Productivity Tip',
    #             'habit': '🔄 Habit Reminder',
    #             'health': '💚 Health Nudge',
    #             'general': '🔔 GUM Nudge'
    #         }
            
    #         title = title_map.get(notification_type, '🔔 GUM Nudge')
    #         escaped_title = title.replace('"', '\\"')
            
    #         # Create the AppleScript command
    #         script = f'''
    #         display notification "{escaped_message}" with title "{escaped_title}" sound name "Glass"
    #         '''
            
    #         # Execute the notification
    #         subprocess.run(['osascript', '-e', script], check=True, timeout=5)
    #         self.logger.info(f"📢 Displayed notification: {title} - {message}")
            
    #     except subprocess.CalledProcessError as e:
    #         self.logger.error(f"❌ Failed to display notification: {e}")
    #     except subprocess.TimeoutExpired:
    #         self.logger.error("❌ Notification display timed out")
    #     except Exception as e:
    #         self.logger.error(f"❌ Error displaying notification: {e}")
    
    # --- that ends the initial notification display banner WITHOUT the buttons

    # ADAPTIVE NUDGE ENGINE METHODS
    
    def get_adaptive_nudge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptive nudge engine."""
        if not self.adaptive_nudge_enabled:
            return {"enabled": False}
        
        try:
            obs_stats = self.observation_manager.get_statistics()
            training_stats = self.observation_manager.training_logger.get_statistics()
            
            return {
                "enabled": True,
                "active_observations": obs_stats,
                "training_data": training_stats
            }
        except Exception as e:
            self.logger.error(f"Error getting adaptive nudge statistics: {e}")
            return {"enabled": True, "error": str(e)}
    
    def enable_adaptive_nudge(self) -> None:
        """Enable the adaptive nudge engine."""
        self.adaptive_nudge_enabled = True
        self.logger.info("Adaptive nudge engine enabled")
    
    def disable_adaptive_nudge(self) -> None:
        """Disable the adaptive nudge engine."""
        self.adaptive_nudge_enabled = False
        self.logger.info("Adaptive nudge engine disabled")
    
    def get_active_observations(self) -> Dict[str, Any]:
        """Get currently active observations."""
        if not self.adaptive_nudge_enabled:
            return {}
        
        try:
            return self.observation_manager.get_active_observations()
        except Exception as e:
            self.logger.error(f"Error getting active observations: {e}")
            return {}
    
    def cancel_observation(self, nudge_id: str) -> bool:
        """Cancel an active observation."""
        if not self.adaptive_nudge_enabled:
            return False
        
        try:
            return self.observation_manager.cancel_observation(nudge_id)
        except Exception as e:
            self.logger.error(f"Error cancelling observation {nudge_id}: {e}")
            return False
    
    # ========== ADD ALL THESE NEW METHODS HERE ==========
    
    # this is the notif display with buttons, but modal pop up window
    # def _display_notification_with_buttons(self, message: str, notification_type: str, nudge_id: str):
    #     """Display a native macOS notification with feedback buttons."""
    #     try:
    #         escaped_message = message.replace('"', '\\"').replace("'", "\\'")
            
    #         title_map = {
    #             'break': '⏰ Break Reminder',
    #             'focus': '🎯 Focus Nudge', 
    #             'productivity': '⚡ Productivity Tip',
    #             'habit': '🔄 Habit Reminder',
    #             'health': '💚 Health Nudge',
    #             'general': '🔔 GUM Nudge'
    #         }
            
    #         title = title_map.get(notification_type, '🔔 GUM Nudge')
    #         escaped_title = title.replace('"', '\\"').replace("'", "\\'")
            
    #         # AppleScript with buttons
    #         script = f'''
    #         try
    #             set theResponse to button returned of (display alert "{escaped_title}" message "{escaped_message}" buttons {{"Thanks!", "Not now!"}} default button "Thanks!" giving up after 30)
    #             return theResponse
    #         on error
    #             return "no_response"
    #         end try
    #         '''
            
    #         # Execute and capture button click
    #         def show_alert_async():
    #             try:
    #                 result = subprocess.run(
    #                     ['osascript', '-e', script], 
    #                     capture_output=True, 
    #                     text=True, 
    #                     timeout=35
    #                 )
                    
    #                 response = result.stdout.strip()
    #                 if "Thanks!" in response:
    #                     feedback = "thanks"
    #                 elif "Not now!" in response:
    #                     feedback = "not_now"
    #                 else:
    #                     feedback = "no_response"
                    
    #                 self.button_feedback[nudge_id] = feedback
    #                 self.session_stats[f"{feedback}_count"] += 1
                    
    #                 self.logger.info(f"📊 User feedback for {nudge_id}: {feedback}")
    #                 self._update_satisfaction_multiplier()
                    
    #             except subprocess.TimeoutExpired:
    #                 self.button_feedback[nudge_id] = "no_response"
    #                 self.session_stats["no_response_count"] += 1
    #                 self.logger.info(f"⏱️ No response for {nudge_id} (timeout)")
    #             except Exception as e:
    #                 self.logger.error(f"❌ Error in alert: {e}")
    #                 self.button_feedback[nudge_id] = "no_response"
    #                 self.session_stats["no_response_count"] += 1
            
    #         import threading
    #         thread = threading.Thread(target=show_alert_async, daemon=True)
    #         thread.start()
            
    #         self.logger.info(f"📢 Displayed notification: {title}")
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error displaying notification: {e}")

    # this uses PyObjC to display the notification with buttons as a banner!!! 
    def _display_notification_with_buttons(self, message: str, notification_type: str, nudge_id: str):
        """Display a native macOS notification banner with feedback buttons."""
        try:
            # Try to use NSUserNotification for banner with buttons
            if not PYOJBC_AVAILABLE:
                # Fallback to simple banner without buttons
                self.logger.warning("PyObjC not available - using simple banner notification")
                self._display_simple_notification(message, notification_type)
                
                # Auto-mark as no_response since we can't get feedback
                self.button_feedback[nudge_id] = "no_response"
                self.session_stats["no_response_count"] += 1
                return
            
            # Create notification
            notification = NSUserNotification.alloc().init()
            
            # Set title and message
            title_map = {
                'break': '⏰ Break Reminder',
                'focus': '🎯 Focus Nudge', 
                'productivity': '⚡ Productivity Tip',
                'habit': '🔄 Habit Reminder',
                'health': '💚 Health Nudge',
                'general': '🔔 GUM Nudge'
            }
            title = title_map.get(notification_type, '🔔 GUM Nudge')
            
            notification.setTitle_(title)
            notification.setInformativeText_(message)
            notification.setSoundName_("Glass")
            
            # Add action buttons
            notification.setHasActionButton_(True)
            notification.setActionButtonTitle_("Thanks!")
            notification.setOtherButtonTitle_("Not now!")
            
            # Store nudge_id in userInfo for callback
            notification.setUserInfo_({"nudge_id": nudge_id})
            
            # Set up delegate (only create once)
            if not hasattr(self, '_notification_delegate'):
                self._notification_delegate = NotificationDelegate.alloc().init()
                self._notification_delegate.notifier = self
                
                center = NSUserNotificationCenter.defaultUserNotificationCenter()
                center.setDelegate_(self._notification_delegate)
            
            # Deliver notification
            center = NSUserNotificationCenter.defaultUserNotificationCenter()
            center.deliverNotification_(notification)
            
            self.logger.info(f"📢 Displayed banner notification with buttons: {title}")
            
            # Set default feedback after 30 seconds
            def set_default_feedback():
                import time
                time.sleep(30)
                if nudge_id not in self.button_feedback:
                    self.button_feedback[nudge_id] = "no_response"
                    self.session_stats["no_response_count"] += 1
                    self.logger.info(f"⏱️ No response for {nudge_id} (timeout)")
            
            import threading
            thread = threading.Thread(target=set_default_feedback, daemon=True)
            thread.start()
                
        except Exception as e:
            self.logger.error(f"❌ Error displaying notification: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to simple notification
            try:
                self._display_simple_notification(message, notification_type)
                self.button_feedback[nudge_id] = "no_response"
                self.session_stats["no_response_count"] += 1
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback notification also failed: {fallback_error}")

    def _display_simple_notification(self, message: str, notification_type: str):
        """Fallback: Display simple banner notification without buttons."""
        try:
            escaped_message = message.replace('"', '\\"')
            
            title_map = {
                'break': '⏰ Break Reminder',
                'focus': '🎯 Focus Nudge', 
                'productivity': '⚡ Productivity Tip',
                'habit': '🔄 Habit Reminder',
                'health': '💚 Health Nudge',
                'general': '🔔 GUM Nudge'
            }
            
            title = title_map.get(notification_type, '🔔 GUM Nudge')
            escaped_title = title.replace('"', '\\"')
            
            script = f'''
            display notification "{escaped_message}" with title "{escaped_title}" sound name "Glass"
            '''
            
            subprocess.run(['osascript', '-e', script], check=True, timeout=5)
            self.logger.info(f"📢 Displayed banner notification: {title}")
            
        except Exception as e:
            self.logger.error(f"❌ Error displaying simple notification: {e}")
    
    def _update_satisfaction_multiplier(self):
        """Update cooldown/threshold multiplier based on recent satisfaction."""
        recent_feedback = list(self.button_feedback.values())[-10:]
        
        if len(recent_feedback) < 3:
            return
        
        thanks_count = recent_feedback.count("thanks")
        not_now_count = recent_feedback.count("not_now")
        total_responses = thanks_count + not_now_count
        
        if total_responses == 0:
            satisfaction_rate = 0.5
        else:
            satisfaction_rate = thanks_count / total_responses
        
        if satisfaction_rate < 0.3:
            self.satisfaction_multiplier = 2.0
            self.logger.warning(
                f"😟 Low satisfaction rate ({satisfaction_rate:.1%}) - "
                f"backing off (multiplier: {self.satisfaction_multiplier:.1f}x)"
            )
        elif satisfaction_rate < 0.5:
            self.satisfaction_multiplier = 1.5
            self.logger.info(
                f"😐 Moderate satisfaction rate ({satisfaction_rate:.1%}) - "
                f"reducing frequency (multiplier: {self.satisfaction_multiplier:.1f}x)"
            )
        elif satisfaction_rate > 0.7:
            self.satisfaction_multiplier = 1.0
            self.logger.info(
                f"😊 High satisfaction rate ({satisfaction_rate:.1%}) - "
                f"maintaining pace (multiplier: {self.satisfaction_multiplier:.1f}x)"
            )
        else:
            self.satisfaction_multiplier = 0.9
            self.logger.info(
                f"🙂 Good satisfaction rate ({satisfaction_rate:.1%}) - "
                f"normal operation (multiplier: {self.satisfaction_multiplier:.1f})"
            )
    
    def get_satisfaction_score(self) -> float:
        """Calculate current session satisfaction score."""
        thanks = self.session_stats["thanks_count"]
        not_now = self.session_stats["not_now_count"]
        total_responses = thanks + not_now
        
        if total_responses == 0:
            return 0.5
        
        return thanks / total_responses
    
    def _get_satisfaction_status(self, satisfaction_rate: float) -> str:
        """Get human-readable satisfaction status."""
        if satisfaction_rate < 0.3:
            return "low_satisfaction_backing_off"
        elif satisfaction_rate < 0.5:
            return "moderate_satisfaction_reducing_frequency"
        elif satisfaction_rate > 0.7:
            return "high_satisfaction_maintaining_pace"
        else:
            return "good_satisfaction_normal_operation"
    
    def save_session_report(self):
        """Save session statistics to a timestamped JSON file."""
        try:
            session_dir = Path(self.gum_instance._data_directory) / "session_reports"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_slug = self.user_name.lower().replace(' ', '_')
            filename = f"session_report_{user_slug}_{timestamp}.json"
            filepath = session_dir / filename
            
            satisfaction_rate = self.get_satisfaction_score()
            total = self.session_stats['total_notifications']
            
            report_data = {
                "session_metadata": {
                    "user_name": self.user_name,
                    "start_time": self.session_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": (datetime.now() - self.session_start_time).total_seconds()
                },
                "notification_stats": {
                    "total_notifications": total,
                    "thanks_count": self.session_stats['thanks_count'],
                    "not_now_count": self.session_stats['not_now_count'],
                    "no_response_count": self.session_stats['no_response_count']
                },
                "rates": {
                    "thanks_rate": self.session_stats['thanks_count'] / max(1, total),
                    "not_now_rate": self.session_stats['not_now_count'] / max(1, total),
                    "no_response_rate": self.session_stats['no_response_count'] / max(1, total),
                    "satisfaction_score": satisfaction_rate
                },
                "adaptive_behavior": {
                    "final_multiplier": self.satisfaction_multiplier,
                    "status": self._get_satisfaction_status(satisfaction_rate)
                },
                "notification_details": [
                    {
                        "nudge_id": notif.get('nudge_id', 'unknown'),
                        "timestamp": notif['timestamp'],
                        "type": notif['type'],
                        "message": notif['message'],
                        "scores": {
                            "relevance": notif['relevance'],
                            "goal_relevance": notif.get('goal_relevance'),
                            "urgency": notif['urgency'],
                            "impact": notif['impact']
                        },
                        "button_feedback": self.button_feedback.get(notif.get('nudge_id', ''), 'no_response')
                    }
                    for notif in self.sent_notifications
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"💾 Session report saved to: {filepath}")
            print(f"\n{'='*60}")
            print(f"📊 SESSION REPORT SAVED")
            print(f"{'='*60}")
            print(f"Location: {filepath}")
            print(f"Satisfaction Score: {satisfaction_rate:.1%}")
            print(f"Total Notifications: {total}")
            print(f"{'='*60}\n")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving session report: {e}")
            return None
    # ===================================================
    
    def _is_in_cooldown(self) -> tuple[bool, float]:
        """
        Check if we're in cooldown period since last notification.
        
        Returns:
            tuple: (is_in_cooldown, seconds_remaining)
        """
        if self.last_notification_time is None:
            return False, 0.0
        
        # Apply satisfaction multiplier to cooldown
        adaptive_cooldown = self.min_notification_interval * self.satisfaction_multiplier
        
        time_since_last = (datetime.now() - self.last_notification_time).total_seconds()
        
        if time_since_last < adaptive_cooldown:
            remaining = adaptive_cooldown - time_since_last
            self.logger.info(
                f"Cooldown active: {time_since_last:.0f}s since last notification "
                f"({remaining:.0f}s remaining, adaptive cooldown: {adaptive_cooldown:.0f}s)"
            )
            return True, remaining
        
        return False, 0.0