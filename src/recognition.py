from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "kmewhort/beit-sketch-classifier"

# Load processor and model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict_sketch(image_path: str):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred_idx = logits.argmax(dim=-1).item()
    label = model.config.id2label[pred_idx]
    confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()

    return label, confidence
