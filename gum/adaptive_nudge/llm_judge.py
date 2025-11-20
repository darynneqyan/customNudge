#!/usr/bin/env python3
"""
LLM Judge Module for Adaptive Nudge Engine

Evaluates whether a user acted on a nudge based on post-nudge system state
using a specialized LLM prompt designed for impartial evaluation.

Goal: Provide accurate, consistent evaluation of nudge effectiveness
through structured LLM analysis of user behavior patterns.
"""

import json
import logging
from typing import Dict, Any, Tuple, List
import os

from ..providers import create_provider

class LLMJudge:
    """
    LLM-based judge for evaluating nudge effectiveness.
    
    This class uses a specialized LLM prompt to evaluate whether a user
    acted on a nudge based on their post-nudge system state.
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the LLM judge.
        
        Args:
            model: LLM model to use for judgment (defaults to env var or gpt-4o-mini)
            api_key: API key for LLM provider (defaults to env var)
        """
        self.logger = logging.getLogger("gum.adaptive_nudge.llm_judge")
        
        # Use environment variables or defaults
        model = model or "gemini-2.5-flash"
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Initialize LLM provider
        self.provider = create_provider(
            model=model,
            api_key=api_key
        )
        
        self.logger.info(f"LLMJudge initialized with model: {model}")
        
        # Prompt for evaluating system state snapshots
        self.system_eval_prompt = """You are an expert in human-computer interaction. Evaluate whether a user's system state aligns with a nudge goal.

# Task
Evaluate the system state snapshot to determine if the user's activity aligns with the nudge's goal.

# User Goal (from nudge)
{nudge_content}

# System State Snapshot
- Active Application: {active_app}
- Window Title: {window_title}
- Browser Tabs: {browser_tabs}
- Open Files: {open_files}
- Recent Applications: {recent_apps}
- System Info: {system_info}
- Clipboard: {clipboard}

# Evaluation
Determine if this snapshot shows goal-aligned behavior based on the nudge's goal.

Respond with:
GOAL_ALIGNED: [true or false]
REASONING: [brief explanation]"""

        # Prompt for evaluating observations when system state is ambiguous
        self.observations_prompt = """You are an expert in human-computer interaction. Evaluate whether a user acted on a nudge based on observations.

# Task
You will be given:
1. A nudge message (user goal)
2. System state classification results (compliance percentage and pattern)
3. A sequence of observations during the 2-minute window

# User Goal (from nudge)
{nudge_content}

# System State Classification Results
- Compliance Percentage: {compliance_percentage:.1f}%
- Pattern: {pattern}

# Observations During 2-Minute Window
{observations}

# Evaluation
Since system state is {pattern}, use observations to determine final score:
- If Partially Compliant (30-70%): Determine if observations confirm compliance (Score = 1), non-compliance (Score = 0), or are ambiguous (Score = 0.5)
- If Non-Compliant (<30%): Determine if observations show attempt/compliance (Score = 0.5) or confirm non-compliance (Score = 0)

# Output Format
<reasoning>
[Your reasoning here - analyze observations in context of system state pattern]
</reasoning>

[0 or 0.5 or 1]"""

    async def get_judge_score(
        self, 
        nudge: str, 
        observations: List[Dict[str, Any]] = None,
        post_nudge_system_state_snapshots: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get LLM judgment on nudge effectiveness.
        
        Uses nudge text as user goal. Evaluates system state snapshots first.
        If compliant (>70%), returns 1. Otherwise, uses observations to determine score.
        
        Args:
            nudge: The nudge message (used as user goal)
            observations: List of observations during the window (for pattern analysis)
            post_nudge_system_state_snapshots: List of system state snapshots at regular intervals (30 secs)
            
        Returns:
            Dict containing:
            - score: 0 (ineffective), 0.5 (partially effective), or 1 (effective)
            - reasoning: LLM's reasoning for the score
        """
        try:
            self.logger.info(f"Evaluating nudge effectiveness: {nudge[:50]}...")
            
            # Step 1: Evaluate system capture
            compliance_percentage, pattern = await self._classify_snapshots(nudge, post_nudge_system_state_snapshots or [])
            
            # Step 1.1: If compliant (>70%), return 1
            if pattern == "Compliant":
                reasoning = f"System state shows clear compliance ({compliance_percentage:.1f}% goal-aligned snapshots). User acted on the nudge."
                self.logger.info(f"Judge evaluation: score=1 (Compliant), reasoning={reasoning[:100]}...")
                return {
                    "score": 1.0,
                    "reasoning": reasoning,
                    "compliance_percentage": compliance_percentage,
                    "pattern": pattern
                }
            
            # Step 1.2: If not compliant or ambiguous, look to observations
            observations_text = self._format_observations(observations or [])
            
            formatted_prompt = self.observations_prompt.format(
                nudge_content=nudge,
                compliance_percentage=compliance_percentage,
                pattern=pattern,
                observations=observations_text
            )
            
            response = await self.provider.chat_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "text"}
            )
            
            # Step 2: Return 0, 1, or 0.5 based on observations
            score, reasoning = self._parse_judge_response(response)
            
            self.logger.info(f"Judge evaluation: score={score}, reasoning={reasoning[:100]}...")
            
            return {
                "score": score,
                "reasoning": reasoning,
                "compliance_percentage": compliance_percentage,
                "pattern": pattern
            }
            
        except Exception as e:
            self.logger.error(f"Error getting judge score: {e}")
            return {
                "score": 0,  # Default to ineffective on error
                "reasoning": f"Error in evaluation: {str(e)}",
                "compliance_percentage": 0,
                "pattern": "Non-Compliant"
            }
    
    async def _classify_snapshots(self, nudge: str, snapshots: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        Classify system state snapshots using LLM evaluation.
        
        Uses nudge text as user goal. Evaluates each snapshot to determine
        if it shows goal-aligned behavior.
        
        Args:
            nudge: The nudge message (used as user goal)
            snapshots: List of system state snapshots
            
        Returns:
            Tuple of (compliance_percentage, pattern)
        """
        if not snapshots:
            return 0.0, "Non-Compliant"
        
        # Evaluate each snapshot using LLM
        goal_aligned_count = 0
        for snapshot in snapshots:
            is_aligned = await self._evaluate_snapshot(nudge, snapshot)
            if is_aligned:
                goal_aligned_count += 1
        
        # Calculate compliance percentage
        compliance_percentage = (goal_aligned_count / len(snapshots)) * 100
        
        # Classify pattern
        if compliance_percentage > 70:
            pattern = "Compliant"
        elif compliance_percentage >= 30:
            pattern = "Partially Compliant"
        else:
            pattern = "Non-Compliant"
        
        return compliance_percentage, pattern
    
    async def _evaluate_snapshot(self, nudge: str, snapshot: Dict[str, Any]) -> bool:
        """
        Evaluate a single snapshot to determine if it's goal-aligned.
        
        Args:
            nudge: The nudge message (used as user goal)
            snapshot: System state snapshot
            
        Returns:
            True if snapshot shows goal-aligned behavior, False otherwise
        """
        try:
            # Extract system capture variables
            active_app = snapshot.get('active_app', 'Unknown')
            window_title = snapshot.get('window_title', 'Unknown')
            browser_tabs = json.dumps(snapshot.get('browser_tabs', []), indent=2)
            open_files = json.dumps(snapshot.get('open_files', []), indent=2)
            recent_apps = json.dumps(snapshot.get('recent_apps', []), indent=2)
            system_info = json.dumps(snapshot.get('system_info', {}), indent=2)
            clipboard = json.dumps(snapshot.get('clipboard_history', []), indent=2)
            
            # Format prompt with system capture variables
            formatted_prompt = self.system_eval_prompt.format(
                nudge_content=nudge,
                active_app=active_app,
                window_title=window_title,
                browser_tabs=browser_tabs,
                open_files=open_files,
                recent_apps=recent_apps,
                system_info=system_info,
                clipboard=clipboard
            )
            
            response = await self.provider.chat_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "text"}
            )
            
            # Parse response to determine if goal-aligned
            if "GOAL_ALIGNED: true" in response.upper() or "GOAL_ALIGNED:true" in response.upper():
                return True
            elif "GOAL_ALIGNED: false" in response.upper() or "GOAL_ALIGNED:false" in response.upper():
                return False
            else:
                # Default to False if unclear
                self.logger.warning(f"Unclear response from snapshot evaluation: {response[:100]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating snapshot: {e}")
            return False
    
    def _format_observations(self, observations: List[Dict[str, Any]]) -> str:
        """
        Format observations for the prompt.
        
        Args:
            observations: List of observation dictionaries
            
        Returns:
            Formatted string of observations
        """
        if not observations:
            return "No observations available during the window."
        
        formatted = []
        for i, obs in enumerate(observations, 1):
            timestamp = obs.get('timestamp', obs.get('created_at', 'Unknown'))
            content = obs.get('content', '')[:200]  # Limit length
            formatted.append(f"Time {timestamp}: {content}...")
        
        return "\n".join(formatted)
    
    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        """
        Parse the LLM judge response to extract score and reasoning.
        
        This method handles various response formats and extracts the
        numerical score (0, 0.5, or 1) and reasoning text from the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Tuple of (score, reasoning) where score is 0, 0.5, or 1
        """
        try:
            # Extract reasoning from <reasoning> tags
            reasoning_start = response.find("<reasoning>")
            reasoning_end = response.find("</reasoning>")
            
            if reasoning_start != -1 and reasoning_end != -1:
                reasoning = response[reasoning_start + 11:reasoning_end].strip()
            else:
                # Fallback: look for reasoning in the response
                reasoning = "No structured reasoning provided"
                if len(response) > 200:
                    reasoning = response[:200] + "..."
                else:
                    reasoning = response
            
            # Extract score (0, 0.5, or 1) - look for the last occurrence
            lines = response.strip().split('\n')
            score = None
            
            # Look for score in the last few lines
            for line in reversed(lines[-5:]):
                line = line.strip()
                if line in ['0', '0.5', '1']:
                    score = float(line)
                    break
                # Also check for "0.5" as separate tokens
                if line == '0.5' or line == '.5':
                    score = 0.5
                    break
            
            # Fallback: look for score anywhere in response
            if score is None:
                # Check for 0.5 first (most specific)
                if '0.5' in response or '.5' in response:
                    score = 0.5
                elif '1' in response and '0' not in response and '0.5' not in response:
                    score = 1.0
                elif '0' in response and '0.5' not in response:
                    score = 0.0
                else:
                    # Default to 0 if unclear
                    score = 0.0
                    reasoning = f"Unclear response, defaulting to 0. Response: {response[:100]}..."
            
            return score, reasoning
                
        except Exception as e:
            self.logger.error(f"Error parsing judge response: {e}")
            return 0.0, f"Parse error: {str(e)}"
    
    async def batch_evaluate(self, nudge_data_pairs: list) -> list:
        """
        Evaluate multiple nudge-snapshot pairs in batch.
        
        This method can be used for batch processing of multiple
        evaluations, potentially improving efficiency.
        
        Args:
            nudge_data_pairs: List of (nudge, observations, snapshots) tuples
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for nudge, observations, snapshots in nudge_data_pairs:
            try:
                result = await self.get_judge_score(nudge, observations, snapshots)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch evaluation: {e}")
                results.append({
                    "score": 0,
                    "reasoning": f"Batch evaluation error: {str(e)}",
                    "compliance_percentage": 0,
                    "pattern": "Non-Compliant"
                })
        
        return results
