DeepFake Image Detection

Overview
This repository contains the code for training and evaluating a deepfake image detection model using **EfficientNet-B3**. The model is fine-tuned to classify images as either **real** or **fake**, leveraging **PyTorch** and **Torchvision**.

Repository Structure
```
├── training.py        # Script to train the deepfake detection model
├── evaluation.py      # Script to evaluate the model on new images and generate JSON predictions
├── README.md          # Project documentation
```

Dataset Structure
Ensure that your dataset is organized as follows:
```
Dataset/
├── real/              # Folder containing real images
├── fake/              # Folder containing deepfake images
```
For evaluation:
```
eval/
├── image1.jpg
├── image2.jpg
...
```

Training the Model
1. Install dependencies:
   pip install torch torchvision matplotlib json
2. Run the training script:
   python training.py
   - The model will train using EfficientNet-B3 as the backbone.
   - Dataset is split into 80% training and 20% validation.
   - Augmentation techniques such as horizontal flips, rotations, and color jitter are applied.
   - Training checkpoints and logs are stored.

## Evaluating the Model
1. Place test images inside the 'eval/' folder.
2. Run the evaluation script:
   python evaluation.py
3. The script will:
   - Load the trained model.
   - Predict the class of each image (real or fake).
   - Generate a JSON file (`predictions.json`) in the following format:
   ```json
   [
       {"index": 1, "prediction": "fake"},
       {"index": 2, "prediction": "real"},
       ...
   ]
   ```
   

Model Choice: Why EfficientNet-B3?
- Performance vs. Efficiency: Provides a strong accuracy-efficiency balance.
- Pre-trained Features: Utilizes ImageNet knowledge for robust feature extraction.
- Better Generalization: Works well on deepfake images beyond just human faces.
- Scalability: Allows further fine-tuning for different datasets.

Challenges & Solutions
- CUDA Errors: Resolved device-side asserts by adjusting model transfers.
- Corrupted Images: Ensured proper image loading and conversion to RGB.
- Label Mapping Issues: Verified dataset folder structure and batch labels.
- Data Imbalance: Maintained balanced class distributions across training and validation.

Future Improvements
- Explore additional deepfake detection architectures.
- Implement adversarial training to improve robustness.
- Deploy as a web application for real-time deepfake detection.

Contributors
Koushik Balaji P – Model training, evaluation, and documentation.

License
This project is open-source and available under the MIT License.

