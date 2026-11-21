import logging
import os
import urllib.parse
from pathlib import Path
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

logger = logging.getLogger(__name__)

# Directory for generated images
GENERATED_IMAGE_DIR = Path("generated_images")
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _save_image(image: Image.Image) -> str:
    """Helper to save PIL image object and return normalized string path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    image_path = GENERATED_IMAGE_DIR / f"generated_{timestamp}.png"
    image.save(image_path)
    logger.info(f"[Image Generator] Image saved: {image_path}")
    return str(image_path).replace("\\", "/")


def generate_image(prompt: str) -> str | None:
    """
    Generate realistic, high-quality, watermark-free images.
    
    Order of Providers:
    1. Cloudflare Workers AI (SDXL Lightning / SDXL Base) -> Clean, free 8k photorealism
    2. Hugging Face (FLUX.1-schnell) -> If HF account has credits
    3. Pollinations API -> Free fallback
    """

    # Enhance prompt for photorealism and crisp detail
    enhanced_prompt = f"{prompt}, raw realistic photograph, highly detailed, sharp focus, 8k resolution, studio lighting, natural colors"
    logger.info(f"[Image Generator] Generating image for prompt: '{prompt}'")

    # ---------------------------------------------------------
    # 1. Primary Option: Cloudflare Workers AI (SDXL Lightning & Base)
    # ---------------------------------------------------------
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if account_id and api_token:
        cf_models = [
            "@cf/bytedance/stable-diffusion-xl-lightning",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0"
        ]
        headers = {"Authorization": f"Bearer {api_token}"}
        
        for model in cf_models:
            try:
                logger.info(f"[Image Generator] Requesting image from Cloudflare ({model.split('/')[-1]})...")
                url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
                response = requests.post(
                    url,
                    headers=headers,
                    json={"prompt": enhanced_prompt, "num_steps": 20},
                    timeout=30
                )
                if response.status_code == 200 and len(response.content) > 1000:
                    image = Image.open(BytesIO(response.content))
                    return _save_image(image)
                else:
                    logger.warning(f"[Image Generator] Cloudflare {model} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"[Image Generator] Cloudflare model {model} failed: {e}")

    # ---------------------------------------------------------
    # 2. Secondary Option: Hugging Face (FLUX.1-schnell)
    # ---------------------------------------------------------
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            logger.info("[Image Generator] Requesting image from Hugging Face...")
            client = InferenceClient(token=token)
            image = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )
            if image:
                return _save_image(image)
        except Exception as e:
            logger.warning(f"[Image Generator] Hugging Face failed ({e})")

    # ---------------------------------------------------------
    # 3. Tertiary Fallback: Pollinations API
    # ---------------------------------------------------------
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
        logger.info("[Image Generator] Requesting image from Pollinations API...")
        
        response = requests.get(pollinations_url, timeout=30)
        if response.status_code == 200 and len(response.content) > 1000:
            image = Image.open(BytesIO(response.content))
            return _save_image(image)
    except Exception as e:
        logger.error(f"[Image Generator] Pollinations fallback failed: {str(e)}")

    return None


