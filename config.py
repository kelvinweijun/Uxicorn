import os, time
from cerebras.cloud.sdk import Cerebras
from stores import api_key_state

def load_api_keys():
    """Load all API keys from environment and config file"""
    keys = []
    
    # Try environment variable first
    env_key = os.environ.get("CEREBRAS_API_KEY")
    if env_key:
        keys.append(env_key.strip())
    
    # Load from config.txt (supports multiple keys, one per line)
    try:
        with open('config.txt', 'r') as f:
            for line in f:
                key = line.strip()
                if key and key not in keys:  # Avoid duplicates
                    keys.append(key)
    except:
        pass
    
    return keys

def get_cerebras_client(rotate=False):
    """Get Cerebras client with automatic key rotation on rate limits"""
    global api_key_state
    
    # Load keys if not already loaded
    if not api_key_state['keys']:
        api_key_state['keys'] = load_api_keys()
    
    if not api_key_state['keys']:
        raise ValueError("CEREBRAS_API_KEY not found. Please set it in environment variables or create a config.txt file with your API key(s), one per line.")
    
    # Rotate to next key if requested
    if rotate and len(api_key_state['keys']) > 1:
        api_key_state['current_index'] = (api_key_state['current_index'] + 1) % len(api_key_state['keys'])
        api_key_state['last_rotation'] = time.time()
        print(f"🔄 Rotated to API key #{api_key_state['current_index'] + 1}/{len(api_key_state['keys'])}")
    
    current_key = api_key_state['keys'][api_key_state['current_index']]
    return Cerebras(api_key=current_key)

def call_cerebras_with_retry(client_func, max_retries=None):
    """
    Wrapper to call Cerebras API with automatic retry and key rotation on rate limits.
    
    Args:
        client_func: Function that takes a Cerebras client and makes an API call
        max_retries: Maximum number of retries (defaults to number of API keys)
    
    Returns:
        API response
    """
    global api_key_state
    
    if max_retries is None:
        max_retries = len(api_key_state['keys']) if api_key_state['keys'] else 1
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            client = get_cerebras_client(rotate=(attempt > 0))
            return client_func(client)
        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Check if it's a rate limit error
            if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                print(f"⚠️  Rate limit hit on API key #{api_key_state['current_index'] + 1}")
                
                if attempt < max_retries - 1:
                    print(f"🔄 Trying next API key... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"❌ All {max_retries} API keys exhausted")
            else:
                # Non-rate-limit error, don't retry
                raise
    
    # If we get here, all retries failed
    raise last_error
