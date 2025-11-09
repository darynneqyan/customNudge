#!/usr/bin/env python3
"""
Monitor batch processing status
"""
import os
import time
import sqlite3
from pathlib import Path

def check_queue_size():
    """Check the current queue size"""
    queue_dir = Path.home() / ".cache" / "gum" / "batches" / "queue"
    if not queue_dir.exists():
        return 0
    
    # Count files in queue directory
    try:
        files = list(queue_dir.iterdir())
        return len([f for f in files if f.is_file()])
    except Exception as e:
        print(f"Error checking queue: {e}")
        return 0

def check_database_activity():
    """Check recent database activity"""
    db_path = Path.home() / ".cache" / "gum" / "gum.db"
    if not db_path.exists():
        return "No database found"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check recent observations
        cursor.execute("SELECT COUNT(*) FROM observations WHERE created_at > datetime('now', '-1 hour')")
        recent_obs = cursor.fetchone()[0]
        
        # Check recent propositions
        cursor.execute("SELECT COUNT(*) FROM propositions WHERE created_at > datetime('now', '-1 hour')")
        recent_props = cursor.fetchone()[0]
        
        conn.close()
        return f"Recent: {recent_obs} observations, {recent_props} propositions"
    except Exception as e:
        return f"DB error: {e}"

def main():
    print("Monitoring GUM batch processing...")
    print("=" * 50)
    
    last_queue_size = 0
    stable_count = 0
    
    while True:
        queue_size = check_queue_size()
        db_status = check_database_activity()
        
        print(f"[{time.strftime('%H:%M:%S')}] Queue: {queue_size} items | {db_status}")
        
        # Check if queue is stable (not processing)
        if queue_size == last_queue_size and queue_size >= 3:
            stable_count += 1
            if stable_count >= 3:
                print(f"⚠️  WARNING: Queue has been stable at {queue_size} items for {stable_count} checks")
                print("   This suggests batch processing may not be working")
        else:
            stable_count = 0
            
        last_queue_size = queue_size
        time.sleep(10)

if __name__ == "__main__":
    main()
