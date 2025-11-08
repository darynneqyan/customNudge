from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import os
import argparse
import asyncio
import shutil  
from gum import gum
from gum.observers import Screen
from gum.goal import prompt_for_goal

class QueryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, '')
        else:
            setattr(namespace, self.dest, values)

def parse_args():
    parser = argparse.ArgumentParser(description='GUM - A Python package with command-line interface')
    parser.add_argument('--user-name', '-u', type=str, help='The user name to use')
    
    parser.add_argument(
        '--query', '-q',
        nargs='?',
        action=QueryAction,
        help='Query the GUM with an optional query string',
    )
    
    parser.add_argument('--limit', '-l', type=int, help='Limit the number of results', default=10)
    parser.add_argument('--model', '-m', type=str, help='Model to use (e.g., gpt-4o-mini, gemini-1.5-pro)')
    parser.add_argument('--reset-cache', action='store_true', help='Reset the GUM cache and exit')  # Add this line
    
    # Batching configuration arguments
    parser.add_argument('--min-batch-size', type=int, help='Minimum number of observations to trigger batch processing')
    parser.add_argument('--max-batch-size', type=int, help='Maximum number of observations per batch')
    
    # Notification configuration
    parser.add_argument('--enable-notifications', action='store_true', help='Enable notification processing')

    args = parser.parse_args()

    if not hasattr(args, 'query'):
        args.query = None

    return args

async def main(args, user_goal=None):
    model = args.model or os.getenv('MODEL_NAME') or 'gpt-4o-mini'
    user_name = args.user_name or os.getenv('USER_NAME')

    # Batching configuration - follow same pattern as other args    
    min_batch_size = args.min_batch_size or int(os.getenv('MIN_BATCH_SIZE', '5'))
    max_batch_size = args.max_batch_size or int(os.getenv('MAX_BATCH_SIZE', '15'))

    # you need one or the other
    if user_name is None and args.query is None:
        print("Please provide a user name (as an argument, -u, or as an env variable) or a query (as an argument, -q)")
        return
    
    # Query Mode
    if args.query is not None:
        gum_instance = gum(user_name, model)
        await gum_instance.connect_db()
        result = await gum_instance.query(args.query, limit=args.limit)
        
        # confidences / propositions / number of items returned
        print(f"\nFound {len(result)} results:")
        for prop, score in result:
            print(f"\nProposition: {prop.text}")
            if prop.reasoning:
                print(f"Reasoning: {prop.reasoning}")
            if prop.confidence is not None:
                print(f"Confidence: {prop.confidence:.2f}")
            print(f"Relevance Score: {score:.2f}")
            print("-" * 80)
    
    # Listening Mode 
    else:
        print(f"Listening to {user_name} with model {model}")
        
        # Check if notifications are enabled
        enable_notifications = args.enable_notifications or os.getenv('ENABLE_NOTIFICATIONS', '').lower() == 'true'
        
        if enable_notifications:
            print("Notifications: ENABLED")
            
        async with gum(
            user_name, 
            model, 
            Screen(model),
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            enable_notifications=enable_notifications,
            user_goal=user_goal
        ) as gum_instance:
            try:
                await asyncio.Future()  # run forever (Ctrl-C to stop)
            except asyncio.CancelledError:
                print("\nShutting down...")
                raise

def cli():
    args = parse_args()
    
    # Handle --reset-cache before anything else (synchronously)
    if getattr(args, 'reset_cache', False):
        cache_dir = os.path.expanduser('~/.cache/gum/')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Deleted cache directory: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Determine if we're in listening mode or query mode 
    user_name = args.user_name or os.getenv('USER_NAME')
    is_listening_mode = args.query is None and user_name is not None
    
    user_goal = None
    if is_listening_mode:
        print(f"Listening to {user_name} with model {args.model or os.getenv('MODEL_NAME') or 'gpt-4o-mini'}")
        print("Prompting for your goal...")
        user_goal = prompt_for_goal()
        
        if user_goal is None:
            print("No goal entered. Starting without a specific goal.")
            user_goal = None
        else:
            print(f"Goal set: {user_goal}")
    
    try:
        asyncio.run(main(args, user_goal=user_goal))
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == '__main__':
    cli()