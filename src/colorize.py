# src/colorize.py

import cv2
import os

def colorize_any(input_path, output_path=None, colormap=cv2.COLORMAP_JET):
    """
    Colorize a grayscale image using a colormap.
    
    Args:
        input_path (str): Path to the input grayscale image.
        output_path (str, optional): Path to save the colorized image.
                                     Defaults to 'datasets/colorized_output.png'.
        colormap (OpenCV Colormap, optional): OpenCV colormap to apply. Default is COLORMAP_JET.
        
    Returns:
        str: Path to the colorized image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    # Read image as grayscale
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read image: {input_path}")

    # Apply OpenCV colormap
    colored = cv2.applyColorMap(gray, colormap)

    # Set default output path if not provided
    if output_path is None:
        os.makedirs("datasets", exist_ok=True)
        output_path = "datasets/colorized_output.png"

    # Save colorized image
    cv2.imwrite(output_path, colored)
    print(f"Colorized image saved at: {output_path}")

    return output_path


if __name__ == "__main__":
    # Example usage
    colorize_any("datasets/gray_output.png")
