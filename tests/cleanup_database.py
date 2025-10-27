#!/usr/bin/env python3
"""
Clean up corrupted database and restart fresh
"""

import os
import shutil
from pathlib import Path

def cleanup_database():
    """Clean up the corrupted database and restart fresh"""
    print("🧹 Cleaning up corrupted database...")
    
    cache_dir = Path.home() / '.cache/gum'
    
    if cache_dir.exists():
        # Backup the corrupted database
        backup_dir = cache_dir.parent / 'gum_backup'
        backup_dir.mkdir(exist_ok=True)
        
        print(f"📦 Backing up corrupted database to {backup_dir}")
        if (cache_dir / 'gum.db').exists():
            shutil.copy2(cache_dir / 'gum.db', backup_dir / 'gum.db.corrupted')
        
        # Remove the corrupted database files
        print("🗑️  Removing corrupted database files...")
        for file in cache_dir.glob('gum.db*'):
            print(f"   Removing {file}")
            file.unlink()
        
        # Keep the batches directory but clear it
        batches_dir = cache_dir / 'batches'
        if batches_dir.exists():
            print("🗑️  Clearing batches directory...")
            shutil.rmtree(batches_dir)
            batches_dir.mkdir()
        
        # Keep screenshots directory but clear it
        screenshots_dir = cache_dir / 'screenshots'
        if screenshots_dir.exists():
            print("🗑️  Clearing screenshots directory...")
            shutil.rmtree(screenshots_dir)
            screenshots_dir.mkdir()
        
        print("✅ Database cleanup complete!")
        print("🔄 You can now restart GUM with a fresh database")
        
    else:
        print("❌ No cache directory found")

if __name__ == "__main__":
    cleanup_database()
