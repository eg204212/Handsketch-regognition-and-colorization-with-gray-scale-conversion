#main.py

from recognition import predict_sketch
from grayscale import convert_to_grayscale
from colorize import simple_colorize

if _name_ == "_main_":
    image_path = "datasets/sample_sketch2.png"
    gray_path = "datasets/gray_output.png"
    color_path = "outputs/gray_output_colorized.png"

    print("[1] Converting to Grayscale...")
    convert_to_grayscale(image_path, gray_path)

    print("[2] Recognizing Sketch...")
    label, confidence = predict_sketch(image_path)
    print(f"Predicted Label: {label} (Confidence: {confidence:.2f}%)")

    print("[3] Colorizing Sketch with AI Generation...")
from colorize import colorize_any
colorized_path = colorize_any(gray_path, label, output_path=color_path)
print(f"Generated image saved at: {colorized_path}")