#!/usr/bin/env python3
"""
Simple test to trace batch processing flow.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gum import gum
from gum.observers import Screen

async def simple_test():
    """Simple test of GUM creation."""
    
    print("=== SIMPLE GUM TEST ===\n")
    
    # Set up environment
    os.environ['GOOGLE_API_KEY'] = "AIzaSyBKYsQ77txAwduIif62yQPanVx2QYG7CuE"
    
    # Create GUM instance exactly like CLI does
    print("Creating GUM instance...")
    gum_instance = gum(
        "test_user",
        "gemini-2.5-flash",
        Screen("gemini-2.5-flash"),
        min_batch_size=3,
        max_batch_size=10,
        enable_notifications=True
    )
    
    print("GUM instance created successfully!")
    print(f"User: {gum_instance.user_name}")
    print(f"Model: {gum_instance.model}")
    print(f"Min batch size: {gum_instance.min_batch_size}")
    print(f"Max batch size: {gum_instance.max_batch_size}")
    print(f"Notifications enabled: {gum_instance.enable_notifications}")

if __name__ == "__main__":
    asyncio.run(simple_test())
