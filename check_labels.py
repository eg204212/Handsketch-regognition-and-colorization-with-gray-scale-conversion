from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "kmewhort/beit-sketch-classifier"

# Load processor and model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# Print all possible categories
print("Categories this model can predict:")
for idx, label in model.config.id2label.items():
    print(f"{idx}: {label}")
