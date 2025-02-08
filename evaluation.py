import os
import json
from PIL import Image
import torch

# Path to your evaluation folder containing 500 images
eval_folder = 'give your file path'

# List all image files in the eval folder; sort them to ensure consistent ordering
eval_image_files = sorted([f for f in os.listdir(eval_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(eval_image_files)} images for evaluation.")

# Prepare a list to store the prediction results
results = []

# Loop over each image, apply the validation transform, run inference, and record the prediction.
# (Assumes that 'data_transforms' contains a 'val' transform defined previously and that
# full_dataset.classes is available. In our previous code, full_dataset.classes was ['fake', 'real'].)
model_ft.eval()
with torch.no_grad():
    for idx, img_filename in enumerate(eval_image_files, start=1):
        img_path = os.path.join(eval_folder, img_filename)
        try:
            # Open the image and convert to RGB (in case it's grayscale)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_filename}: {e}")
            continue

        # Apply the validation transform and add a batch dimension
        input_tensor = data_transforms['val'](image).unsqueeze(0).to(device)

        # Forward pass to get outputs
        outputs = model_ft(input_tensor)
        _, pred = torch.max(outputs, 1)

        # Map predicted index to label; in our dataset, index 0 is 'fake', index 1 is 'real'
        predicted_label = full_dataset.classes[pred.item()]

        results.append({
            "index": idx,
            "prediction": predicted_label
        })

# Save the results to a JSON file
output_json_path = '/content/eval_predictions.json'
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"JSON file generated at: {output_json_path}")
