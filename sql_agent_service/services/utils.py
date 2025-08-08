from google.api_core.exceptions import ResourceExhausted 
import asyncio

async def call_with_retry(func, *args, **kwargs):
    """Calls a synchronous function with exponential backoff on ResourceExhausted errors."""
    max_retries = 5
    delay = 1.0  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            # Run the synchronous SDK call in a separate thread
            return await asyncio.to_thread(func, *args, **kwargs)
        except ResourceExhausted as e:
            if attempt < max_retries - 1:
                print(f"Quota exceeded. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Double the delay for the next retry
            else:
                print("Max retries reached. Failing.")
                raise e
            
