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
from typing import Dict, Any, Tuple
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
        model = model or os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GUM_LM_API_KEY")
        
        # Initialize LLM provider
        self.provider = create_provider(
            model=model,
            api_key=api_key
        )
        
        self.logger.info(f"LLMJudge initialized with model: {model}")
        
        # High-quality prompt for the LLM Judge
        self.judge_prompt = """You are an expert in human-computer interaction and behavioral psychology. Your task is to determine if a user acted on a given nudge based on their post-nudge system state.

# Task
You will be given:
1. A nudge message that was sent to the user
2. A "screenshot" of their system state after the nudge (captured 3 minutes later)

Your job is to evaluate whether the user's actions suggest they followed the nudge's suggestion.

# Evaluation Criteria

**Score 1 (Effective) if:**
- The user's system state shows evidence they acted on the nudge's suggestion
- Their current activity aligns with what the nudge recommended
- There are clear behavioral changes that correspond to the nudge's intent
- The user appears to have engaged with the suggested action

**Score 0 (Ineffective) if:**
- The user's system state shows no evidence of following the nudge
- Their current activity is unrelated to the nudge's suggestion
- The user appears to have ignored or dismissed the nudge
- No behavioral changes are evident that correspond to the nudge

# Important Guidelines

1. **Be Conservative**: Only score as "Effective" if there's clear evidence of action
2. **Consider Context**: Look for subtle changes that might indicate compliance
3. **Avoid False Positives**: Don't assume unrelated activities are responses to the nudge
4. **Focus on Intent**: Consider whether the user's actions align with the nudge's goal
5. **Time Sensitivity**: Remember this was captured 3 minutes after the nudge
6. **Application Context**: Consider what applications the user is using and whether they align with the nudge
7. **Web Activity**: Look at browser tabs and URLs for evidence of following suggestions
8. **File Activity**: Consider open files and recent applications for work-related nudges

# Output Format

Provide your reasoning in a <reasoning> tag, then on a new line, provide only:
- 0 (ineffective) 
- 1 (effective)

# Example

<reasoning>
The nudge suggested taking a break from coding, but the user is still actively working in VS Code on the same project. No evidence of following the break suggestion.
</reasoning>
0

# Your Evaluation

Nudge: {nudge_content}

Post-Nudge System State:
- Active Application: {active_app}
- Window Title: {window_title}
- Browser Tabs: {browser_tabs}
- Open Files: {open_files}
- Recent Applications: {recent_apps}
- System Info: {system_info}
- Clipboard: {clipboard}

<reasoning>
[Your detailed reasoning here]
</reasoning>
[0 or 1]"""

    async def get_judge_score(self, nudge: str, screenshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM judgment on nudge effectiveness.
        
        This is the main method that formats the nudge and system state,
        sends it to the LLM, and parses the response to extract the score
        and reasoning.
        
        Args:
            nudge: The nudge message that was sent to the user
            screenshot: Post-nudge system state dictionary
            
        Returns:
            Dict containing:
            - score: 0 (ineffective) or 1 (effective)
            - reasoning: LLM's reasoning for the score
        """
        try:
            self.logger.info(f"Evaluating nudge effectiveness: {nudge[:50]}...")
            
            # Format the prompt with actual data
            formatted_prompt = self.judge_prompt.format(
                nudge_content=nudge,
                active_app=screenshot.get('active_app', 'Unknown'),
                window_title=screenshot.get('window_title', 'Unknown'),
                browser_tabs=json.dumps(screenshot.get('browser_tabs', []), indent=2),
                open_files=json.dumps(screenshot.get('open_files', []), indent=2),
                recent_apps=json.dumps(screenshot.get('recent_apps', []), indent=2),
                system_info=json.dumps(screenshot.get('system_info', {}), indent=2),
                clipboard=json.dumps(screenshot.get('clipboard_history', []), indent=2)
            )
            
            # Call LLM
            response = await self.provider.chat_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "text"}
            )
            
            # Parse response
            score, reasoning = self._parse_judge_response(response)
            
            self.logger.info(f"Judge evaluation: score={score}, reasoning={reasoning[:100]}...")
            
            return {
                "score": score,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.logger.error(f"Error getting judge score: {e}")
            return {
                "score": 0,  # Default to ineffective on error
                "reasoning": f"Error in evaluation: {str(e)}"
            }
    
    def _parse_judge_response(self, response: str) -> Tuple[int, str]:
        """
        Parse the LLM judge response to extract score and reasoning.
        
        This method handles various response formats and extracts the
        numerical score (0 or 1) and reasoning text from the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Tuple of (score, reasoning) where score is 0 or 1
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
            
            # Extract score (0 or 1) - look for the last occurrence
            lines = response.strip().split('\n')
            score = None
            
            # Look for score in the last few lines
            for line in reversed(lines[-5:]):
                line = line.strip()
                if line in ['0', '1']:
                    score = int(line)
                    break
            
            # Fallback: look for score anywhere in response
            if score is None:
                if '1' in response and '0' not in response:
                    score = 1
                elif '0' in response:
                    score = 0
                else:
                    # Default to 0 if unclear
                    score = 0
                    reasoning = f"Unclear response, defaulting to 0. Response: {response[:100]}..."
            
            return score, reasoning
                
        except Exception as e:
            self.logger.error(f"Error parsing judge response: {e}")
            return 0, f"Parse error: {str(e)}"
    
    async def batch_evaluate(self, nudge_screenshot_pairs: list) -> list:
        """
        Evaluate multiple nudge-screenshot pairs in batch.
        
        This method can be used for batch processing of multiple
        evaluations, potentially improving efficiency.
        
        Args:
            nudge_screenshot_pairs: List of (nudge, screenshot) tuples
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for nudge, screenshot in nudge_screenshot_pairs:
            try:
                result = await self.get_judge_score(nudge, screenshot)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch evaluation: {e}")
                results.append({
                    "score": 0,
                    "reasoning": f"Batch evaluation error: {str(e)}"
                })
        
        return results
