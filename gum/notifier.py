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
import uuid

from .db_utils import search_propositions_bm25, search_observations_bm25, get_related_propositions
from .models import Observation, Proposition
from .prompts.gum import NOTIFICATION_DECISION_PROMPT
from .adaptive_nudge import ObservationWindowManager

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
    goal_relevance_score: Optional[float] 
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
        
        # Get project root directory (where this file is located: gum/notifier.py -> gum -> project_root)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # gum/notifier.py -> gum -> project_root
        
        # Use absolute paths to ensure files are always in project root, regardless of working directory
        self.contexts_file = project_root / f"notification_contexts_{user_name.lower().replace(' ', '_')}.json"
        self.decisions_file = project_root / f"notification_decisions_{user_name.lower().replace(' ', '_')}.json"
        
        self.logger.info(f"Notifier file paths - contexts: {self.contexts_file}, decisions: {self.decisions_file}")
        
        # Initialize Adaptive Nudge Engine
        self.observation_manager = ObservationWindowManager(user_name, gum_instance=self.gum_instance, notifier=self)
        self.adaptive_nudge_enabled = True  # Can be controlled via environment variable
        
        # Load existing contexts if available
        self._load_notification_log()
        # Load existing decisions if available
        self._load_decisions_log()

        # Cooldown configuration
        self.last_notification_time = None
        self.min_notification_interval = 120 # 2 minutes
        
        # Button feedback and session tracking
        self.button_feedback: Dict[str, str] = {}
        self.session_stats = {
            "thanks_count": 0,
            "not_now_count": 0,
            "no_response_count": 0,
            "total_notifications": 0
        }
        self.satisfaction_multiplier = 1.0
        self.session_start_time = datetime.now()
    
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
    
    def _load_decisions_log(self):
        """Load notification decisions from file."""
        if self.decisions_file.exists():
            try:
                with open(self.decisions_file, 'r') as f:
                    self.decisions_log = json.load(f)
                if len(self.decisions_log) > 0:
                    self.logger.info(f"Loaded {len(self.decisions_log)} existing notification decisions")
                else:
                    self.logger.info("No existing notification decisions found - system will create new decisions as observations are processed")
            except Exception as e:
                self.logger.error(f"Error loading decisions log: {e}")
                self.decisions_log = []
        else:
            self.logger.info("Notification decisions file not found - system will create new decisions as observations are processed")
    
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
                'goal_relevance_score': decision.goal_relevance_score,
                'urgency_score': decision.urgency_score,
                'impact_score': decision.impact_score,
                'reasoning': decision.reasoning,
                'notification_message': decision.notification_message,
                'notification_type': decision.notification_type,
                'observation_content': context.observation_content[:200] + "...",
                'generated_propositions_count': len(context.generated_propositions),
                'similar_propositions_count': len(context.similar_propositions),
                'similar_observations_count': len(context.similar_observations),
                'blocked_reason': None  # Will be set if blocked by cooldown
            }
            
            # Verify reasoning is present before saving
            if not decision.reasoning or len(decision.reasoning.strip()) == 0:
                self.logger.error(f"Attempting to save decision with empty reasoning! Observation ID: {context.observation_id}")
            
            self.decisions_log.append(decision_entry)
            
            self.logger.info(f"Saving decision to {self.decisions_file}")
            with open(self.decisions_file, 'w') as f:
                json.dump(self.decisions_log, f, indent=2)
                f.flush()  # Ensure file is written to OS buffer immediately
            self.logger.info(f"Decision saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving decision: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_decision_with_cooldown(self, context: NotificationContext, remaining: float):
        """Update the most recent decision entry with cooldown information."""
        try:
            if self.decisions_log:
                # Update the most recent decision entry
                last_decision = self.decisions_log[-1]
                if last_decision.get('observation_id') == context.observation_id:
                    last_decision['blocked_reason'] = 'COOLDOWN'
                    last_decision['cooldown_remaining'] = remaining
                    last_decision['should_notify'] = False  # Mark as not sent due to cooldown
                    
                    # Save updated decision
                    with open(self.decisions_file, 'w') as f:
                        json.dump(self.decisions_log, f, indent=2)
                        f.flush()  # Ensure file is written to OS buffer immediately
                    self.logger.info(f"Updated decision with cooldown information")
        except Exception as e:
            self.logger.error(f"Error updating decision with cooldown: {e}")
    
    def update_decision_with_effectiveness(self, nudge_id: str, judge_score: Dict[str, Any]):
        """
        Update a decision entry with effectiveness evaluation results.
        
        Args:
            nudge_id: The nudge ID to match
            judge_score: Dictionary containing:
                - score: 0, 0.5, or 1
                - reasoning: LLM's reasoning
                - compliance_percentage: Percentage of goal-aligned snapshots
                - pattern: "Compliant", "Partially Compliant", or "Non-Compliant"
        """
        try:
            # Find decision entry by nudge_id
            for decision in self.decisions_log:
                if decision.get('nudge_id') == nudge_id:
                    decision['effectiveness_score'] = judge_score.get('score', 0)
                    decision['effectiveness_reasoning'] = judge_score.get('reasoning', '')
                    decision['compliance_percentage'] = judge_score.get('compliance_percentage', 0)
                    decision['compliance_pattern'] = judge_score.get('pattern', 'Unknown')
                    decision['evaluation_source'] = 'system_capture' if judge_score.get('pattern') == 'Compliant' else 'observations'
                    
                    # Save updated decision
                    with open(self.decisions_file, 'w') as f:
                        json.dump(self.decisions_log, f, indent=2)
                        f.flush()  # Ensure file is written to OS buffer immediately
                    self.logger.info(f"Updated decision {nudge_id} with effectiveness data")
                    return
            
            self.logger.warning(f"Could not find decision entry with nudge_id: {nudge_id}")
        except Exception as e:
            self.logger.error(f"Error updating decision with effectiveness: {e}")
    
    
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

        # Get user goal from gum_instance or default 
        user_goal = getattr(self.gum_instance, 'user_goal', None)
        if user_goal:
            user_goal_text = f"{self.user_name} has set the following goal for this session: {user_goal}"
        else:
            user_goal_text = f"{self.user_name} has not set a specific goal for this session. Consider general behavioral patterns and what they likely want to improve."
        
        # Get cooldown status for LLM
        in_cooldown, remaining = self._is_in_cooldown()
        if in_cooldown:
            cooldown_status = f"⚠️ COOLDOWN ACTIVE: A notification was sent {remaining:.0f} seconds ago. You MUST set `should_notify` to `false` and explain in your reasoning that the reason is cooldown - too soon since last notification."
        else:
            cooldown_status = "✅ Cooldown inactive: No notification sent in the last 2 minutes. Ready to send if observation is relevant."
        
        # Construct prompt
        prompt = NOTIFICATION_DECISION_PROMPT.format(
            user_name=self.user_name,
            user_goal=user_goal_text,
            observation_content=observation_content,
            generated_propositions=gen_props_text,
            similar_propositions=sim_props_text,
            similar_observations=sim_obs_text,
            notification_history=notif_history_text,
            learning_context=learning_context,
            cooldown_status=cooldown_status
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
            goal_relevance = result.get('goal_relevance_score')
            goal_relevance_score = float(goal_relevance) if goal_relevance is not None else None
            
            reasoning = result.get('reasoning', '')
            if not reasoning:
                self.logger.warning("LLM response has no 'reasoning' field! Response keys: " + str(list(result.keys())))
                reasoning = "No reasoning provided by LLM"
            
            decision = NotificationDecision(
                should_notify=result.get('should_notify', False),
                relevance_score=float(result.get('relevance_score', 0)),
                goal_relevance_score=goal_relevance_score,
                urgency_score=float(result.get('urgency_score', 0)),
                impact_score=float(result.get('impact_score', 0)),
                reasoning=reasoning,
                notification_message=result.get('notification_message', ''),
                notification_type=result.get('notification_type', 'none')
            )
            
            goal_relevance = f"{decision.goal_relevance_score}/10" if decision.goal_relevance_score is not None else "N/A (no goal)"
            self.logger.info(
                f"LLM Decision: should_notify={decision.should_notify}, "
                f"relevance={decision.relevance_score}, goal_relevance={goal_relevance}, "
                f"urgency={decision.urgency_score}, impact={decision.impact_score}"
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
                
                # Use LLM to decide whether to notify (timeout protects against slow API responses)
                try:
                    decision = await asyncio.wait_for(
                        self._make_notification_decision(
                            observation_content,
                            generated_props_dict,
                            similar_propositions,
                            similar_observations
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self.logger.error(
                        f"⚠️ LLM decision timed out for observation {observation_id}; skipping notification"
                    )
                    decision = None
                
                if decision:
                    self.logger.info(f"LLM made a decision: should_notify={decision.should_notify}")
                    # Save decision to file for GUI
                    self._save_decision(context, decision)
                    
                    # Log decision
                    goal_relevance = f"{decision.goal_relevance_score}/10" if decision.goal_relevance_score is not None else "N/A (no goal)"
                    self.logger.info(
                        f"LLM Decision - Should notify: {decision.should_notify}, "
                        f"Relevance: {decision.relevance_score}/10, "
                        f"Goal Relevance: {goal_relevance}, "
                        f"Urgency: {decision.urgency_score}/10, "
                        f"Impact: {decision.impact_score}/10"
                    )
                    
                    # Check if LLM mentioned cooldown in reasoning
                    reasoning_lower = decision.reasoning.lower()
                    is_cooldown_reason = 'cooldown' in reasoning_lower or 'too soon' in reasoning_lower or 'recently sent' in reasoning_lower
                    
                    # Print to console
                    print(f"\n{'='*60}")
                    print(f"LLM NOTIFICATION DECISION")
                    print(f"{'='*60}")
                    print(f"Should Notify: {'✅ YES' if decision.should_notify else '❌ NO'}")
                    print(f"Type: {decision.notification_type}")
                    print(f"Relevance: {decision.relevance_score}/10")
                    goal_relevance = f"{decision.goal_relevance_score}/10" if decision.goal_relevance_score is not None else "N/A (no goal)"
                    print(f"Goal Relevance: {goal_relevance}")
                    print(f"Urgency: {decision.urgency_score}/10")
                    print(f"Impact: {decision.impact_score}/10")
                    print(f"Reasoning: {decision.reasoning}")
                    
                    # If LLM mentioned cooldown as reason, display it prominently
                    if is_cooldown_reason and not decision.should_notify:
                        print(f"\n⏱️  REASON: COOLDOWN - LLM decided not to notify due to cooldown period")
                    
                    if decision.should_notify:
                        # Safety check: if LLM says should_notify=True but we're in cooldown, block it
                        in_cooldown, remaining = self._is_in_cooldown()
                        if in_cooldown:
                            print(f"\n⚠️  WARNING: LLM said should_notify=True but cooldown is active. Blocking notification.")
                            print(f"Cooldown remaining: {remaining:.0f}s")
                            print(f"{'='*60}\n")
                            # Update saved decision with cooldown reason
                            self._update_decision_with_cooldown(context, remaining)
                            continue
                        print(f"\n📢 NOTIFICATION MESSAGE:")
                        print(f"{decision.notification_message}")
                        print(f"{'='*60}\n")
                        
                        # previously:
                        # # Display native macOS notification
                        # self._display_notification(decision.notification_message, decision.notification_type)

                        nudge_id = str(uuid.uuid4())
                        
                        # Update cooldown timer
                        self.last_notification_time = datetime.now()
                        
                        # Track sent notification (before displaying, so it's logged even if display fails)
                        self.sent_notifications.append({
                            'nudge_id': nudge_id,
                            'timestamp': context.timestamp,
                            'message': decision.notification_message,
                            'type': decision.notification_type,
                            'relevance': decision.relevance_score,
                            'goal_relevance': decision.goal_relevance_score,
                            'urgency': decision.urgency_score,
                            'impact': decision.impact_score
                        })
                        
                        # Display notification asynchronously (non-blocking)
                        # Start the notification task but don't wait for it - batch processing continues immediately
                        asyncio.create_task(
                            self._display_notification_with_swift_async(
                                decision.notification_message,
                                decision.notification_type,
                                nudge_id
                            )
                        )
                        self.logger.info(f"📢 Notification task started for nudge {nudge_id} (non-blocking)")
                        
                        # ADAPTIVE NUDGE ENGINE: Start observation window
                        if self.adaptive_nudge_enabled:
                            try:
                                nudge_data = {
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
                                    'observation_duration': 120,  # 2 minutes
                                    'decision_timestamp': context.timestamp,  # For matching decision entry
                                    'decision_file': str(self.decisions_file)  # For updating decision entry
                                }
                                
                                # Start observation window asynchronously
                                nudge_id = await self.observation_manager.start_observation(nudge_data)
                                
                                # Add nudge_id to the most recent decision entry
                                if self.decisions_log:
                                    last_decision = self.decisions_log[-1]
                                    if last_decision.get('timestamp') == context.timestamp:
                                        last_decision['nudge_id'] = nudge_id
                                        # Save updated decision
                                        with open(self.decisions_file, 'w') as f:
                                            json.dump(self.decisions_log, f, indent=2)
                                            f.flush()  # Ensure file is written to OS buffer immediately
                                
                                self.logger.info(f"Started adaptive nudge observation window with nudge_id: {nudge_id}")
                                
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
    
    def _get_swift_notifier_path(self) -> Path:
        """Get the path to the Swift notifier binary."""
        # Get the directory where this file is located
        current_file = Path(__file__).resolve()
        # Navigate to project root (gum/notifier.py -> gum -> project_root)
        project_root = current_file.parent.parent
        swift_notifier_dir = project_root / "notifier" / "swift_notifier"
        
        # Try app bundle first (preferred for macOS recognition)
        app_bundle = swift_notifier_dir / "GUM Notifier.app"
        app_binary = app_bundle / "Contents" / "MacOS" / "GUM Notifier"
        if app_binary.exists():
            return app_binary
        
        # Fallback to standalone binary (symlink or direct)
        standalone_binary = swift_notifier_dir / "mac_notifier"
        if standalone_binary.exists():
            return standalone_binary
        
        # Return the app bundle path even if it doesn't exist (for error messages)
        return app_binary
    
    def _start_notification_task(self, message: str, notification_type: str, nudge_id: str):
        """
        Start a notification task in the background without blocking.
        
        This method immediately starts the notification display process and returns,
        allowing batch processing to continue. Feedback is collected asynchronously.
        
        Args:
            message: The notification message content
            notification_type: Type of notification (break, focus, etc.)
            nudge_id: Unique identifier for this nudge
        """
        try:
            # Get the current event loop - must be called from async context
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - cannot use async fallback, just log and record
            self.logger.warning(f"No event loop available for async notification - skipping notification")
            self.logger.warning(f"Notification would have been: {message[:50]}...")
            self._record_feedback(nudge_id, "no_response")
            return
        
        # Create and fire off the background task - don't await it
        task = loop.create_task(
            self._display_notification_with_swift_async(message, notification_type, nudge_id)
        )
        
        # Add error handler to catch any unhandled exceptions
        def handle_task_done(task):
            """Handle task completion and log any errors."""
            try:
                task.result()  # This will raise if task had an exception
            except Exception as e:
                self.logger.error(f"Unhandled exception in notification task for {nudge_id}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Record failure as no_response
                self._record_feedback(nudge_id, "no_response")
        
        task.add_done_callback(handle_task_done)
    
    async def _display_notification_with_swift_async(self, message: str, notification_type: str, nudge_id: str):
        """
        Display a native macOS notification with interactive buttons using Swift binary.
        This runs asynchronously and collects feedback in the background.
        
        Args:
            message: The notification message content
            notification_type: Type of notification (break, focus, etc.)
            nudge_id: Unique identifier for this nudge
        """
        try:
            swift_binary = self._get_swift_notifier_path()
            
            self.logger.info(f"🔍 Checking Swift binary at: {swift_binary}")
            self.logger.info(f"   Binary exists: {swift_binary.exists()}")
            
            if not swift_binary.exists():
                self.logger.error(f"❌ FALLBACK TRIGGERED: Swift binary not found at {swift_binary}")
                self.logger.error("Please run: cd notifier/swift_notifier && ./build.sh")
                # Fallback to async notification (non-blocking)
                await self._display_notification_async(message, notification_type)
                self._record_feedback(nudge_id, "no_response")
                return
            
            # Customize title based on notification type
            title_map = {
                'break': '⏰ Break Reminder',
                'focus': '🎯 Focus Nudge', 
                'productivity': '⚡ Productivity Tip',
                'habit': '🔄 Habit Reminder',
                'health': '💚 Health Nudge',
                'general': '🔔 GUM Nudge'
            }
            
            title = title_map.get(notification_type, '🔔 GUM Nudge')
            
            # Start the Swift process - don't wait for it here
            process = await asyncio.create_subprocess_exec(
                str(swift_binary), title, message,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Launch feedback collection in background - this is where we wait
            # Use create_task to run feedback collection without blocking
            feedback_task = asyncio.create_task(
                self._collect_swift_feedback(process, nudge_id, message, notification_type)
            )
            
            # Add error handler for feedback collection
            def handle_feedback_error(task):
                try:
                    task.result()
                except Exception as e:
                    self.logger.error(f"Error in feedback collection for {nudge_id}: {e}")
                    self._record_feedback(nudge_id, "no_response")
            
            feedback_task.add_done_callback(handle_feedback_error)
            
        except FileNotFoundError:
            self.logger.error(f"❌ FALLBACK: Swift notifier binary not found. Please build it first.")
            # Fallback to async notification (non-blocking)
            await self._display_notification_async(message, notification_type)
            self._record_feedback(nudge_id, "no_response")
        except Exception as e:
            self.logger.error(f"❌ FALLBACK: Error starting Swift notification for {nudge_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to async notification (non-blocking)
            try:
                await self._display_notification_async(message, notification_type)
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback notification also failed: {fallback_error}")
            self._record_feedback(nudge_id, "no_response")
    
    async def _collect_swift_feedback(self, process: asyncio.subprocess.Process, nudge_id: str, 
                                     message: str, notification_type: str):
        """
        Collect feedback from Swift notification process in the background.
        
        This coroutine waits for the Swift process to complete (user clicks button or timeout),
        parses the response, and records the feedback.
        
        Args:
            process: The subprocess process object
            nudge_id: Unique identifier for this nudge
            message: Notification message (for fallback)
            notification_type: Notification type (for fallback)
        """
        try:
            # Wait for process to complete with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=35.0  # 30 second timeout + buffer
            )
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            returncode = process.returncode
            
            # ALWAYS log what actually happened (for debugging)
            if returncode != 0:
                self.logger.error(f"❌ Swift notifier failed for {nudge_id}")
                self.logger.error(f"   Return code: {returncode}")
                self.logger.error(f"   STDOUT: '{stdout_text.strip()}'")
                self.logger.error(f"   STDERR: '{stderr_text.strip()}'")
                
                # Check for permission errors
                stderr_lower = stderr_text.lower()
                if "permission denied" in stderr_lower or "error 1" in stderr_lower or "unauthorized" in stderr_lower or "not authorized" in stderr_lower:
                    self.logger.error("❌ FALLBACK: Notification permissions not granted!")
                    self.logger.error("   To grant permissions:")
                    self.logger.error("   1. Open System Settings → Notifications & Focus")
                    self.logger.error("   2. Find 'Terminal' or 'Python' in the app list")
                    self.logger.error("   3. Enable 'Allow Notifications'")
                    self.logger.error("   4. Set alert style to 'Banners' or 'Alerts'")
                
                # Trigger fallback
                await self._display_notification_async(message, notification_type)
                self._record_feedback(nudge_id, "no_response")
                return
            
            # Parse successful response
            feedback = stdout_text.strip().lower()
            
            # Validate and record feedback
            if feedback in ["thanks", "not_now", "no_response"]:
                self.logger.info(f"📊 User feedback for {nudge_id}: {feedback}")
                self._record_feedback(nudge_id, feedback)
            else:
                self.logger.warning(f"Unexpected response from Swift notifier: '{feedback}'")
                if stderr_text:
                    self.logger.warning(f"   STDERR: '{stderr_text.strip()}'")
                self._record_feedback(nudge_id, "no_response")
                
        except asyncio.TimeoutError:
            # Kill the process and record timeout
            try:
                process.kill()
                await process.wait()
            except:
                pass
            self.logger.warning(f"⏱️ Notification timeout for {nudge_id}")
            self._record_feedback(nudge_id, "no_response")
        except Exception as e:
            self.logger.error(f"❌ Error collecting Swift feedback for {nudge_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self._record_feedback(nudge_id, "no_response")
    
    def _record_feedback(self, nudge_id: str, feedback: str):
        """
        Record user feedback from notification buttons.
        
        This is a synchronous method that updates feedback storage, session stats,
        and satisfaction multiplier. Called after feedback is collected.
        
        Args:
            nudge_id: Unique identifier for this nudge
            feedback: User feedback ("thanks", "not_now", or "no_response")
        """
        # Store feedback
        self.button_feedback[nudge_id] = feedback
        
        # Update session statistics
        self.session_stats[f"{feedback}_count"] = self.session_stats.get(f"{feedback}_count", 0) + 1
        self.session_stats["total_notifications"] = self.session_stats.get("total_notifications", 0) + 1
        
        # Update satisfaction multiplier based on feedback
        self._update_satisfaction_multiplier()
        
        self.logger.info(f"✅ Recorded feedback for {nudge_id}: {feedback} (total: {self.session_stats['total_notifications']})")
    
    async def test_swift_binary_diagnostics(self):
        """Diagnostic test to see what's actually happening with Swift binary."""
        swift_binary = self._get_swift_notifier_path()
        
        print(f"\n{'='*60}")
        print(f"SWIFT BINARY DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"Binary path: {swift_binary}")
        print(f"Binary exists: {swift_binary.exists()}")
        
        if swift_binary.exists():
            # Test 1: Direct execution from shell
            print(f"\nTest 1: Direct shell execution...")
            result = subprocess.run(
                [str(swift_binary), "Test", "Direct shell test"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"  Return code: {result.returncode}")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            
            # Test 2: Async subprocess (how it's actually used)
            print(f"\nTest 2: Async subprocess execution...")
            process = await asyncio.create_subprocess_exec(
                str(swift_binary), "Test", "Async subprocess test",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
            print(f"  Return code: {process.returncode}")
            print(f"  STDOUT: {stdout.decode()}")
            print(f"  STDERR: {stderr.decode()}")
            
            print(f"{'='*60}\n")
        else:
            print(f"❌ Binary not found - cannot run diagnostics")
            print(f"{'='*60}\n")
    
    async def _display_notification_async(self, message: str, notification_type: str):
        """Display a native macOS notification (async fallback method without buttons)."""
        try:
            # Escape quotes in the message (both single and double)
            escaped_message = message.replace('"', '\\"').replace("'", "\\'")
            
            # Customize title based on notification type
            title_map = {
                'break': '⏰ Break Reminder',
                'focus': '🎯 Focus Nudge', 
                'productivity': '⚡ Productivity Tip',
                'habit': '🔄 Habit Reminder',
                'health': '💚 Health Nudge',
                'general': '🔔 GUM Nudge'
            }
            
            title = title_map.get(notification_type, '🔔 GUM Nudge')
            escaped_title = title.replace('"', '\\"').replace("'", "\\'")
            
            # Create the AppleScript command
            script = f'''
            display notification "{escaped_message}" with title "{escaped_title}" sound name "Glass"
            '''
            
            # Use async subprocess instead of blocking subprocess.run
            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                await asyncio.wait_for(process.communicate(), timeout=5.0)
                self.logger.info(f"📢 Displayed fallback notification: {title}")
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.logger.error("❌ Fallback notification timed out")
                
        except Exception as e:
            self.logger.error(f"❌ Error in fallback notification: {e}")
    
    
    def get_recent_contexts(self, limit: int = 10) -> List[NotificationContext]:
        """Get the most recent notification contexts."""
        return self.notification_contexts[-limit:]
    
    def get_context_by_observation_id(self, observation_id: int) -> Optional[NotificationContext]:
        """Get notification context for a specific observation."""
        for context in reversed(self.notification_contexts):
            if context.observation_id == observation_id:
                return context
        return None
    
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
    
    def _is_in_cooldown(self) -> tuple[bool, float]:
        """
        Check if we're in cooldown period since last notification.
        
        Returns:
            tuple: (is_in_cooldown, seconds_remaining)
        """
        if self.last_notification_time is None:
            return False, 0.0
        
        time_since_last = (datetime.now() - self.last_notification_time).total_seconds()
        
        if time_since_last < self.min_notification_interval:
            remaining = self.min_notification_interval - time_since_last
            self.logger.info(
                f"Cooldown active: {time_since_last:.0f}s since last notification "
                f"({remaining:.0f}s remaining)"
            )
            return True, remaining
        
        return False, 0.0