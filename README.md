# DermaDetect AI
"Your skin's safety, Our prority"

# Explainable Skin Lesion Classification using Deep Learning

An end-to-end deep learning project for **multiclass skin lesion classification** using dermoscopic images.
The project benchmarks multiple modern CNN architectures and integrates **explainable AI (Grad-CAM)** to improve model transparency—an important requirement for real-world medical AI systems.

---

## Problem Statement

Early and accurate detection of skin cancer, especially melanoma, is critical but highly dependent on expert interpretation.
This project explores how **deep learning models** can assist clinicians by automatically classifying skin lesions while also providing **visual explanations** for model decisions.

---

## Models Implemented

The following pretrained models were fine-tuned using transfer learning:

* **ResNet18**
  Lightweight residual network with skip connections. Serves as a strong baseline with low computational cost.

* **EfficientNet**
  Parameter-efficient architecture that balances depth, width, and resolution for high accuracy with fewer parameters—suitable for deployment-oriented scenarios.

* **ConvNeXtV2**
  A state-of-the-art convolutional architecture inspired by vision transformers. Achieved the **best overall performance**, particularly in melanoma detection.

All models were initialized with ImageNet weights and trained under identical conditions to ensure fair comparison.

---

## ⚙️ Training & Evaluation

* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** AdamW
* **Epochs:** 30
* **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
* **Explainability:** Grad-CAM heatmaps to visualize regions influencing predictions

---

## 🏆 Key Results

| Model        | Accuracy | F1-Score | Notes                                             |
| ------------ | -------- | -------- | ------------------------------------------------- |
| ResNet18     | 85%      | 0.84     | Strong on common classes, weaker melanoma recall  |
| EfficientNet | 87%      | 0.86     | Balanced performance and computational efficiency |
| ConvNeXtV2   | **88%**  | **0.87** | Best overall performance and melanoma sensitivity |

ConvNeXtV2 demonstrated superior generalization and explainability, making it the most suitable model for clinical decision-support settings.

---

## Explainable AI

Grad-CAM was used to generate heatmaps highlighting image regions that influenced predictions.
This improves **trust, interpretability, and clinical usability**, aligning the model with real-world medical AI requirements.

---

## Key Files

* **`app.py`**
  Main Streamlit application providing the user interface for uploading dermoscopic images and generating real-time predictions.

* **`best_modelcvn.pth`**
  Trained **ConvNeXtV2** model weights. This model achieved the best overall performance, particularly in melanoma detection, and is used for inference in the app.

* **`efficientnet.ipynb`**
  Jupyter notebook containing the complete training, validation, and evaluation pipeline for the EfficientNet model.

* **`convnextv2(1).ipynb`**
  Jupyter notebook for training and evaluating ConvNeXtV2, including performance analysis and Grad-CAM visualization.

* **`resnet18.ipynb`**
  Jupyter notebook implementing ResNet18 as a baseline model, covering preprocessing, training, and metric evaluation.

---

## Datasets Used

This project uses publicly available and widely adopted dermatology datasets:

* **HAM10000**
  Dermoscopic images across multiple skin lesion classes
  🔗 [https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)

* **ISIC Dataset**
  Benchmark dataset for melanoma and skin lesion analysis
  🔗 [https://github.com/iamnarendrasingh/Melanoma-detection/tree/main/dataset](https://github.com/iamnarendrasingh/Melanoma-detection/tree/main/dataset)

---

## Repository Scope

This repository includes:

* Model training and evaluation pipelines
* Data preprocessing and augmentation
* Explainability using Grad-CAM
* Performance comparison across architectures

## Tech Stack

* Python
* PyTorch
* Torchvision / timm
* NumPy, OpenCV, Matplotlib
* Streamlit
* Grad-CAM

---

## License

This project is licensed under the **MIT License**.

The license applies to all original code in this repository.
Pretrained architectures (ResNet18, EfficientNet, ConvNeXtV2) are used via PyTorch-based libraries and are subject to their respective original licenses.


