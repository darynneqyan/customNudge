#!/usr/bin/env python3
"""
Debug specific notification processing steps
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from gum.gum import gum
from gum.observers.screen import Screen
from gum.db_utils import get_related_propositions

async def test_notification_processing_steps():
    """Test each step of the notification processing pipeline"""
    print("🔍 Testing Notification Processing Steps")
    print("=" * 50)
    
    try:
        async with gum(
            "Eyrin",
            "gemini-2.5-flash",
            Screen("gemini-2.5-flash"),
            min_batch_size=3,
            max_batch_size=10,
            enable_notifications=True
        ) as gum_instance:
            print("✅ GUM instance created successfully")
            
            # Wait for some observations to be collected
            print("⏳ Waiting for observations...")
            await asyncio.sleep(15)
            
            # Check if we have observations in the database
            async with gum_instance._session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT COUNT(*) FROM observations"))
                obs_count = result.scalar()
                print(f"📊 Total observations in DB: {obs_count}")
                
                if obs_count > 0:
                    # Get the latest observation
                    result = await session.execute(text("SELECT id, content FROM observations ORDER BY id DESC LIMIT 1"))
                    latest_obs = result.fetchone()
                    
                    if latest_obs:
                        obs_id, obs_content = latest_obs
                        print(f"📊 Latest observation ID: {obs_id}")
                        print(f"📊 Content preview: {obs_content[:100]}...")
                        
                        # Test step 1: Get related propositions
                        print("\n1. Testing get_related_propositions...")
                        try:
                            propositions = await get_related_propositions(session, obs_id)
                            print(f"✅ Found {len(propositions)} propositions")
                            
                            if propositions:
                                print(f"   First proposition: {propositions[0].text[:100]}...")
                        except Exception as e:
                            print(f"❌ Error getting propositions: {e}")
                        
                        # Test step 2: Test notifier methods
                        if gum_instance.notifier:
                            print("\n2. Testing notifier._find_similar_propositions...")
                            try:
                                similar_props = await gum_instance.notifier._find_similar_propositions(propositions)
                                print(f"✅ Found {len(similar_props)} similar propositions")
                            except Exception as e:
                                print(f"❌ Error finding similar propositions: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            print("\n3. Testing notifier._find_similar_observations...")
                            try:
                                similar_obs = await gum_instance.notifier._find_similar_observations(obs_content)
                                print(f"✅ Found {len(similar_obs)} similar observations")
                            except Exception as e:
                                print(f"❌ Error finding similar observations: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            print("\n4. Testing complete notification decision...")
                            try:
                                # Convert propositions to dict format
                                generated_props_dict = []
                                for prop in propositions:
                                    generated_props_dict.append({
                                        "id": prop.id,
                                        "text": prop.text,
                                        "reasoning": prop.reasoning,
                                        "confidence": prop.confidence,
                                        "decay": prop.decay,
                                        "created_at": prop.created_at.isoformat() if hasattr(prop.created_at, 'isoformat') else str(prop.created_at)
                                    })
                                
                                decision = await gum_instance.notifier._make_notification_decision(
                                    obs_content,
                                    generated_props_dict,
                                    similar_props,
                                    similar_obs
                                )
                                
                                if decision:
                                    print(f"✅ Decision made: should_notify={decision.should_notify}")
                                    print(f"   Reasoning: {decision.reasoning}")
                                else:
                                    print("❌ No decision made")
                                    
                            except Exception as e:
                                print(f"❌ Error making notification decision: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("❌ Notifier not initialized")
                else:
                    print("❌ No observations found in database")
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_notification_processing_steps())
