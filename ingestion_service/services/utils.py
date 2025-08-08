import re
from typing import List
from pypdf import PdfReader, PdfWriter
import os
from google.api_core.exceptions import ResourceExhausted 
import asyncio

def sanitize_name(name: str) -> str:
    """Sanitizes a string to be a valid SQL table/column name."""
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)

def split_pdf(file_path: str, doc_id: str, chunk_dir: str, pages_per_chunk: int = 10) -> List[str]:
    """Splits a PDF into smaller chunks and returns a list of chunk file paths."""
    temp_chunk_paths = []
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    
    for i in range(0, num_pages, pages_per_chunk):
        writer = PdfWriter()
        start_page = i
        end_page = min(i + pages_per_chunk, num_pages)
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        
        chunk_path = os.path.join(chunk_dir, f"{doc_id}_chunk_{i//pages_per_chunk + 1}.pdf")
        writer.write(chunk_path)
        temp_chunk_paths.append(chunk_path)
        
    return temp_chunk_paths

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
            
