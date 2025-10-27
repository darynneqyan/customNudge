#!/usr/bin/env python3
"""
Monitor batch processing with simple queue checking
"""
import os
import time
from pathlib import Path

def check_queue_status():
    """Check queue status"""
    queue_dir = Path.home() / ".cache" / "gum" / "batches" / "queue"
    if not queue_dir.exists():
        return "No queue directory"
    
    files = list(queue_dir.iterdir())
    file_count = len([f for f in files if f.is_file()])
    
    # Check if there's an info file (indicates queue is initialized)
    has_info = any(f.name == "info" for f in files)
    
    # Check for data files (q00000, q00001, etc.)
    data_files = [f for f in files if f.name.startswith("q") and f.name != "info"]
    
    return f"Files: {file_count}, Info: {has_info}, Data files: {len(data_files)}"

def main():
    print("Monitoring GUM batch processing...")
    print("=" * 50)
    
    while True:
        status = check_queue_status()
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] {status}")
        
        # Check if GUM process is running
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'gum.cli'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[{timestamp}] ✅ GUM process running (PID: {result.stdout.strip()})")
            else:
                print(f"[{timestamp}] ❌ GUM process not running")
        except Exception as e:
            print(f"[{timestamp}] Error checking process: {e}")
        
        time.sleep(10)

if __name__ == "__main__":
    main()
