# Plant Disease Detector

A deep learning application that identifies diseases in plant leaves from a photo. Upload an image, get a diagnosis in seconds.

**Live demo → [huggingface.co/spaces/AlgoWitch/plant-disease-detector](https://huggingface.co/spaces/AlgoWitch/plant-disease-detector)**

---

## Overview

This project trains a MobileNetV2 convolutional neural network on the PlantVillage dataset to classify plant leaf images into 38 disease categories across 14 crop types. The trained model is served through a Gradio web interface deployed on Hugging Face Spaces.

The goal was to build a complete end-to-end machine learning project — from raw dataset to a live, publicly accessible web application.

---

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | 99.08% |
| Training images | 54,305 |
| Disease classes | 38 |
| Crop types | 14 |
| Training epochs | 10 |

---

## Supported Crops

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## How It Works

1. A leaf photo is resized to 224×224 and normalized
2. The image is passed through a MobileNetV2 model pretrained on ImageNet
3. The final classification layer is fine-tuned on PlantVillage data
4. The model outputs the top 5 most likely disease classes with confidence scores

---

## Tech Stack

- **Model** — PyTorch, MobileNetV2 (transfer learning)
- **Dataset** — PlantVillage (via Kaggle)
- **Training** — Kaggle Notebooks, T4 GPU
- **Interface** — Gradio
- **Deployment** — Hugging Face Spaces

---

## Training Details

- Pretrained MobileNetV2 backbone with a custom classification head
- Data augmentation: random horizontal/vertical flip, rotation up to 30°, color jitter, random grayscale
- Weighted random sampling to handle class imbalance
- Learning rate scheduler: StepLR with step size 3, gamma 0.5
- Optimizer: Adam, learning rate 0.001
- Batch size: 32

---

## Project Structure

```
plant-disease-detector/
├── app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── class_names.json                # List of 38 disease classes
├── plant_disease_detector.ipynb    # Full training notebook
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/AlgoWitch/plant-disease-detector
cd plant-disease-detector
pip install -r requirements.txt
```

Download the model weights from the [Hugging Face Space](https://huggingface.co/spaces/AlgoWitch/plant-disease-detector) and place `plant_disease_model.pth` in the root directory, then:

```bash
python app.py
```

---

## Limitations

- Performs best on well-lit, close-up photos similar to the PlantVillage dataset
- Real-world phone photos may produce less accurate results due to differences in lighting, angle, and background
- Does not support crops outside the 14 trained categories

---

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) by Abdallah Ali, available on Kaggle under CC BY-NC-SA 4.0.
