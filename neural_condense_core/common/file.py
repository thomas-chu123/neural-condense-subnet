import hf_transfer
import numpy as np
import io
import time
import httpx
import os
from tqdm import tqdm
from ..logger import logger

os.makedirs("tmp", exist_ok=True)


async def load_npy_from_url(url: str, max_size_mb: int = 1024):
    """
    Load a `.npy` file from a URL directly into memory.

    Args:
        url (str): URL of the `.npy` file.
        max_size_mb (int): Maximum allowed file size in megabytes.

    Returns:
        tuple: A tuple containing the loaded data as a NumPy array and an error message (empty if no error occurred).
    """
    try:
        # Check file size using an HTTP HEAD request
        async with httpx.AsyncClient() as client:
            response = await client.head(url)
            if response.status_code != 200:
                return None, f"Failed to fetch file info: HTTP {response.status_code}"

            content_length = int(response.headers.get("content-length", 0))
            max_size_bytes = max_size_mb * 1024 * 1024

            if content_length > max_size_bytes:
                return (
                    None,
                    f"File too large: {content_length / (1024 * 1024):.1f}MB exceeds {max_size_mb}MB limit",
                )

            # Stream directly into memory
            start_time = time.time()
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                buffer = io.BytesIO()
                async for chunk in response.aiter_bytes():
                    buffer.write(chunk)

            end_time = time.time()
            logger.info(f"Time taken to download: {end_time - start_time:.2f} seconds")

            # Load NumPy array from buffer
            buffer.seek(0)
            data = np.load(buffer)
            return data, ""

    except Exception as e:
        return None, str(e)
