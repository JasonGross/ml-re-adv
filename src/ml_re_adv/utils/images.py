import base64
import io
from typing import Optional

from PIL import Image


def image_to_base64(image: Image.Image, format: Optional[str] = "png") -> str:
    # Create a bytes buffer to hold the image data
    image_data = io.BytesIO()

    # Save the image to the buffer in the specified format
    image.save(image_data, format=format)

    # Get the raw bytes from the buffer
    image_data.seek(0)  # Rewind the buffer to the beginning
    image_bytes = image_data.read()

    # Encode the bytes to base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_image
