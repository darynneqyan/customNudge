#!/usr/bin/env python3
"""
Real-time monitor for in-context learning.
Watches the decisions file and shows ICL activity as it happens.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import glob

def monitor_icl(user_name: str = "eyrink", watch_interval: float = 2.0):
    """Monitor ICL in real-time."""
    print("=" * 70)
    print("🔍 Real-Time In-Context Learning Monitor")
    print("=" * 70)
    print(f"Watching for decisions file... (refresh every {watch_interval}s)")
    print("Press Ctrl+C to stop\n")
    
    project_root = Path(__file__).parent
    clean_user = user_name.strip().strip('"').strip("'")
    file_suffix = clean_user.lower().replace(' ', '_')
    
    decisions_file = project_root / f"notification_decisions_{file_suffix}.json"
    decisions_file_quoted = project_root / f'notification_decisions_"{file_suffix}".json'
    
    last_count = 0
    last_decision_time = None
    
    try:
        while True:
            # Try to find the file
            file_to_use = None
            if decisions_file.exists():
                file_to_use = decisions_file
            elif decisions_file_quoted.exists():
                file_to_use = decisions_file_quoted
            else:
                # Try fuzzy match
                pattern = f"notification_decisions_*{file_suffix}*.json"
                matches = glob.glob(str(project_root / pattern))
                if matches:
                    file_to_use = Path(matches[0])
            
            if not file_to_use or not file_to_use.exists():
                print(f"\r⏳ Waiting for file: notification_decisions_{file_suffix}.json...", end="", flush=True)
                time.sleep(watch_interval)
                continue
            
            try:
                with open(file_to_use, 'r') as f:
                    decisions = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"\r⏳ File exists but not readable yet...", end="", flush=True)
                time.sleep(watch_interval)
                continue
            
            current_count = len(decisions)
            
            # Check if new decisions arrived
            if current_count > last_count:
                new_decisions = decisions[last_count:]
                print(f"\n{'='*70}")
                print(f"📊 NEW DECISIONS DETECTED: {len(new_decisions)} new decision(s)")
                print(f"{'='*70}\n")
                
                for i, decision in enumerate(new_decisions, 1):
                    print(f"🔔 Decision #{current_count - len(new_decisions) + i}")
                    print(f"   Timestamp: {decision.get('timestamp', 'Unknown')[:19]}")
                    print(f"   Should notify: {decision.get('should_notify', False)}")
                    
                    # Check ICL fields
                    has_icl = 'policy_version' in decision
                    if has_icl:
                        print(f"   ✅ ICL ENABLED")
                        print(f"      Policy version: {decision.get('policy_version', 'N/A')}")
                        print(f"      Examples available: {decision.get('examples_available_count', 0)}")
                        print(f"      Examples used: {decision.get('examples_used_count', 0)}")
                        if decision.get('examples_used'):
                            ex_ids = decision.get('examples_used', [])
                            print(f"      Example IDs: {ex_ids[:3]}{'...' if len(ex_ids) > 3 else ''}")
                        print(f"      Goal bucket: {decision.get('goal_alignment_bucket', 'none')}")
                        print(f"      Time bucket: {decision.get('time_since_last_nudge_bucket', 'unknown')}")
                        time_since = decision.get('time_since_last_nudge')
                        if time_since is None:
                            time_since_str = "N/A (no previous notification)"
                        else:
                            time_since_str = f"{time_since:.1f}s"
                        print(f"      Time since last: {time_since_str}")
                        print(f"      Frequency context: {decision.get('frequency_context', 0)}")
                        print(f"      Pattern: {decision.get('observation_pattern_summary', 'N/A')[:50]}...")
                    else:
                        print(f"   ❌ ICL NOT ENABLED (old format)")
                        print(f"      This decision was made before ICL was enabled")
                    
                    if decision.get('effectiveness_score') is not None:
                        print(f"   📈 Effectiveness: {decision.get('effectiveness_score', 0)}")
                        print(f"      Compliance: {decision.get('compliance_percentage', 0):.1f}%")
                    
                    print()
                
                last_count = current_count
                last_decision_time = datetime.now()
            
            # Show status
            status_parts = []
            status_parts.append(f"Total decisions: {current_count}")
            
            if decisions:
                icl_enabled = sum(1 for d in decisions if 'policy_version' in d)
                status_parts.append(f"ICL enabled: {icl_enabled}/{current_count}")
                
                effective = sum(1 for d in decisions 
                              if d.get('effectiveness_score', 0) >= 0.7 
                              and d.get('should_notify', False))
                status_parts.append(f"Effective examples: {effective}")
                
                with_examples = sum(1 for d in decisions if d.get('examples_used_count', 0) > 0)
                status_parts.append(f"Used examples: {with_examples}")
            
            status = " | ".join(status_parts)
            print(f"\r{status}", end="", flush=True)
            
            time.sleep(watch_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")

if __name__ == "__main__":
    user_name = sys.argv[1] if len(sys.argv) > 1 else "eyrink"
    monitor_icl(user_name)

