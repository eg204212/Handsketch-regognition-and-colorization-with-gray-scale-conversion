# src/fetch_real_image.py
import os
import cv2
import requests
from ddgs import DDGS
import numpy as np

def fetch_real_image(label: str, output_path=None):
    """
    Fetch a real image for the predicted label using DuckDuckGo.
    Converts it to grayscale and returns the path.
    If no image is found, uses a placeholder.
    """
    os.makedirs("datasets", exist_ok=True)
    if output_path is None:
        output_path = f"datasets/real_{label}.png"

    search_terms = [
        label,
        f"{label} photo",
        f"{label} image",
        f"{label} real"
    ]

    img_url = None
    for term in search_terms:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query=term, max_results=3))
            if results:
                img_url = results[0]["image"]
                break
        except Exception:
            continue

    if not img_url:
        print("Could not fetch real image: No results found.")
        placeholder_path = "datasets/placeholder.png"
        if not os.path.exists(placeholder_path):
            placeholder = 255 * np.ones((400, 400), dtype=np.uint8)
            cv2.imwrite(placeholder_path, placeholder)
        return placeholder_path

    # Download the image
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            temp_path = "datasets/temp.jpg"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            img = cv2.imread(temp_path)
            if img is None:
                raise ValueError("Failed to read downloaded image")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path, gray)
            os.remove(temp_path)
            return output_path
        else:
            raise ValueError("Bad response from image URL")
    except Exception as e:
        print(f"Error downloading image: {e}")
        placeholder_path = "datasets/placeholder.png"
        if not os.path.exists(placeholder_path):
            placeholder = 255 * np.ones((400, 400), dtype=np.uint8)
            cv2.imwrite(placeholder_path, placeholder)
        return placeholder_path
