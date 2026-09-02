# CNN-Based Clothing Image Classification

---

## Project Overview

Built, trained, and evaluated multiple Convolutional Neural Network (CNN) architectures from scratch to classify real-world clothing images into 10 categories. The project used the Kaggle Clothing Dataset (1,000 images subset), which contains real-world RGB photographs with varying backgrounds, poses, and lighting, significantly more challenging than benchmark datasets like Fashion-MNIST.

**Categories classified:** Dress, Hat, Longsleeve, Not sure, Outwear, Pants, Shirt, Shoes, Shorts, T-Shirt

---

## Technical Stack

- **Language:** Python 3.11
- **Deep Learning Framework:** PyTorch 2.x with CUDA (GPU-accelerated on NVIDIA RTX 3080)
- **Libraries:** torchvision, scikit-learn, matplotlib, seaborn, NumPy, pandas, Pillow
- **Environment:** Jupyter Notebook
- **Hardware:** NVIDIA RTX 3080 (10GB VRAM)

---

## Dataset & Preprocessing

- **Source:** Kaggle Clothing Dataset (alexeygrigorev/clothing-dataset) — 1,000 images, perfectly balanced at 100 images per class
- **Preprocessing pipeline:**
  - Loaded all images via PIL, converted to RGB, resized to 128×128 pixels
  - Normalized pixel values to [0, 1] range
  - Stratified train/validation/test split: **70% / 15% / 15%** (700 / 150 / 150 images)
  - Converted to PyTorch tensors with channel-first format (N, C, H, W)
- **Custom PyTorch Dataset class** written to handle on-the-fly transforms and data loading

---

## Models Built (6 Total)

### 1. BasicCNN (Baseline)
- 2 convolutional blocks, each with 2 Conv2D layers + BatchNorm + MaxPool + Dropout(0.25)
- Filter counts: 32 → 64
- GlobalAveragePooling → Dense(256) → Dropout(0.5) → Softmax output
- **85,482 parameters**
- Optimizer: Adam (lr=1e-3) | Loss: CrossEntropyLoss
- EarlyStopping (patience=5) + ReduceLROnPlateau scheduler
- **Test Accuracy: 19.33% | Weighted F1: 0.1633 | Training time: 2.5s**

### 2. AugmentedCNN (Same architecture + data augmentation)
- Identical to BasicCNN but trained with augmented data
- Augmentation pipeline: RandomHorizontalFlip + RandomRotation(45°) + RandomResizedCrop + ColorJitter(brightness)
- Augmentation applied via custom transform passed to PyTorch DataLoader
- **85,482 parameters**
- **Test Accuracy: 16.67% | Weighted F1: 0.1009 | Training time: 6.2s**
- *Notable finding: augmentation hurt performance at this dataset scale (70 images/class too few for aggressive augmentation)*

### 3. Deep_3x3 (4-layer deep, all 3×3 kernels — VGG-style)
- 4 convolutional layers: 32 → 64 → 128 → 256 filters, all 3×3 kernels
- MaxPool after every 2nd layer + Dropout(0.25)
- GlobalAveragePooling → Dense(256) → Softmax
- **458,250 parameters**
- **Test Accuracy: 23.33% | Weighted F1: 0.2225 | Training time: 8.7s**

### 4. Deep_5x5 (4-layer deep, all 5×5 kernels — larger receptive field)
- Same structure as Deep_3x3 but all kernels are 5×5
- **1,147,914 parameters**
- **Test Accuracy: 24.00% | Weighted F1: 0.2060 | Training time: 18.9s**
- **Best overall model by test accuracy**

### 5. Wide_3x3 (4-layer wide, doubled filter counts)
- 4 convolutional layers: 64 → 128 → 256 → 512 filters, all 3×3 kernels
- **1,821,706 parameters** (largest model)
- **Test Accuracy: 23.33% | Weighted F1: 0.2208 | Training time: 16.6s**

### 6. Mixed_ker (4-layer, alternating 3×3 and 5×5 kernels)
- Filters: 32 → 64 → 128 → 256, kernels alternating 3×3 / 5×5 / 3×3 / 5×5
- **1,015,306 parameters**
- **Test Accuracy: 18.67% | Weighted F1: 0.1587 | Training time: 14.0s**

---

## Full Results Table

| Model | Parameters | Train Time | Epochs | Test Accuracy | Weighted F1 |
|---|---|---|---|---|---|
| BasicCNN | 85,482 | 2.5s | 9 | 19.33% | 0.1633 |
| AugmentedCNN | 85,482 | 6.2s | 7 | 16.67% | 0.1009 |
| Deep_3x3 | 458,250 | 8.7s | 14 | 23.33% | 0.2225 |
| **Deep_5x5** | **1,147,914** | **18.9s** | **18** | **24.00%** | **0.2060** |
| Wide_3x3 | 1,821,706 | 16.6s | 10 | 23.33% | 0.2208 |
| Mixed_ker | 1,015,306 | 14.0s | 15 | 18.67% | 0.1587 |

---

## Evaluation & Analysis

### Metrics computed per model:
- Test accuracy, test loss
- Per-class precision, recall, F1-score (via sklearn classification_report)
- Weighted F1-score
- Confusion matrix (visualized with seaborn heatmap)
- Most-confused category pairs identified programmatically

### Key per-class findings (BasicCNN):
- **Best classified:** Shoes (recall 0.133–0.667 across models), Hat — distinctive geometric shapes
- **Worst classified:** Pants (0.000 recall in BasicCNN), Not sure (inherently ambiguous label)
- **Most confused pairs:** Shirt↔Longsleeve, Shirt↔T-Shirt, Shoes↔Shirt — all upper-body garments with similar silhouettes

### Visualizations produced:
- Sample images from each category
- Training/validation accuracy and loss curves per model
- Confusion matrices
- Correct predictions (one per class)
- 10 misclassified examples with true vs. predicted labels
- Augmentation examples (9 variations of same image)
- Bar charts comparing all models on accuracy, F1, and parameter count
- Combined validation curves for all 6 models on one plot

---

## Key Findings & Insights

1. **Depth beats augmentation at small dataset scale.** Four-layer architectures consistently outperformed the two-layer baseline by ~4–5 percentage points. Augmentation with only 70 training images per class was counterproductive — the aggressive transforms (45° rotation, random crop) degraded the training signal before the model could establish stable features.

2. **Kernel size had diminishing returns.** Deep_5x5 (24.00%) only marginally outperformed Deep_3x3 (23.33%) despite having 2.5× more parameters and 2× the training time, suggesting 3×3 kernels are sufficient for local texture features in 128×128 clothing images.

3. **GlobalAveragePooling over Flatten.** Using GAP instead of Flatten reduced the dense head from millions of parameters to ~85K for the BasicCNN without significant accuracy loss, enabling faster training and lower memory footprint.

4. **The "Not sure" class is a structural data quality issue.** This label aggregates ambiguous human-annotated items with no consistent visual pattern, making it reliably the hardest class across all models.

5. **Training efficiency on GPU.** All 6 models trained in under 20 seconds each on an RTX 3080, with total project compute under 70 seconds — demonstrating the significant advantage of GPU acceleration for CNN experimentation.

---

## Architecture Design Decisions

- **BatchNormalization** after every convolutional layer for training stability and faster convergence
- **Dropout(0.25)** after each pooling layer + **Dropout(0.50)** in the dense head for regularization
- **EarlyStopping (patience=5)** with `restore_best_weights` equivalent (PyTorch state dict saving) to prevent overtraining
- **ReduceLROnPlateau** scheduler (factor=0.5, patience=3) for automatic learning rate decay
- **Adam optimizer** (lr=1e-3) for all models — adaptive learning rate well-suited for small datasets
- **CrossEntropyLoss** (includes softmax internally) as classification loss function
- Models saved as `.pth` state dictionaries with documented reload instructions

---

## Deliverables

- Jupyter Notebook (`.ipynb`) with all code, outputs, visualizations, and written analysis
- 6 saved model files (`best_ModelName.pth`) including the best performer (`best_Deep_5x5.pth`)
- 10+ figures saved as PNG (training curves, confusion matrices, sample images, comparison charts)
