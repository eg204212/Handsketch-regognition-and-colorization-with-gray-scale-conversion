import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

# Decide tensor dtype based on device (CPU cannot handle float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Load ControlNet (for sketch-to-image)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch_dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch_dtype
)

# Switch scheduler (fixes IndexError with PNDM on CPU)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Move to device
pipe = pipe.to(device)


def colorize_any(input_path, label, output_path="datasets/colorized_output.png"):
    """
    Generate both colored and grayscale realistic images from a sketch.
    
    Args:
        input_path (str): Path to input sketch image.
        label (str): Predicted label from classifier.
        output_path (str): Path to save the colorized image.
    
    Returns:
        tuple(str, str): Paths to (colorized_image, grayscale_image)
    """
    # Load sketch
    sketch = load_image(input_path)
    print("Sketch size:", sketch.size)

    # Prompt for realistic generation
    prompt = f"A realistic photo of a {label}"

    # Generate image
    result = pipe(
        prompt,
        image=sketch,
        num_inference_steps=50,
        guidance_scale=9.0
    )

    # Save colorized result
    colorized_img = result.images[0]
    colorized_img.save(output_path)
    print(f"Generated color image saved at: {output_path}")

    # Convert to grayscale using OpenCV
    colorized_np = np.array(colorized_img)
    gray_np = cv2.cvtColor(colorized_np, cv2.COLOR_RGB2GRAY)
    gray_img = Image.fromarray(gray_np)

    gray_output_path = output_path.replace(".png", "_gray.png")
    gray_img.save(gray_output_path)
    print(f"Generated grayscale image saved at: {gray_output_path}")

    return output_path, gray_output_path


if __name__ == "__main__":
    test_input = "datasets/sample_sketch.png"
    label = "tree"
    color_path, gray_path = colorize_any(test_input, label)
    print(f"Outputs: {color_path}, {gray_path}")
