import logging
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

logger = logging.getLogger(__name__)

# Directory for generated images
GENERATED_IMAGE_DIR = Path("generated_images")
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def generate_image(prompt: str) -> str | None:
    """
    Generate an image using Hugging Face
    and save it locally.

    Returns:
        Relative image path if successful,
        otherwise None.
    """

    try:
        # -----------------------------------------
        # Get Hugging Face token
        # -----------------------------------------

        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN is not configured in .env"
            )

        logger.info(
            "[Image Generator] Generating image using Hugging Face"
        )

        # -----------------------------------------
        # Create Hugging Face client
        # -----------------------------------------

        client = InferenceClient(
            token=token
        )

        # -----------------------------------------
        # Generate image
        # -----------------------------------------

        image = client.text_to_image(
    prompt=prompt,
    model="black-forest-labs/FLUX.1-schnell"
)

        # -----------------------------------------
        # Create unique filename    
        # -----------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        image_path = (
            GENERATED_IMAGE_DIR
            / f"generated_{timestamp}.png"
        )

        # -----------------------------------------
        # Save image
        # -----------------------------------------

        image.save(image_path)

        logger.info(
            f"[Image Generator] Image saved: {image_path}"
        )

        return str(image_path).replace("\\", "/")

    except Exception as e:

        logger.exception(
            f"[Image Generator] Image generation failed: {str(e)}"
        )

        return None