# 🔬 Skin Cancer Detection System

An AI-powered skin cancer detection system using **ResNet-18** for deep learning feature extraction, **XGBoost** for classification, and **Isolation Forest** for out-of-distribution detection — wrapped in a **Streamlit** web application.

---

## 📌 Table of Contents

- [Features](#-features)
- [Supported Conditions](#-supported-skin-conditions)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Train the Model](#-train-the-model)
- [Run the Web App](#-run-the-web-app)
- [Command Line Inference](#-command-line-inference)
- [Model Evaluation](#-model-evaluation)
- [Disclaimer](#-disclaimer)

---

## 🚀 Features

- Deep learning feature extraction using pre-trained **ResNet-18**
- **XGBoost** classifier for accurate skin cancer type prediction
- **Isolation Forest** for out-of-distribution / non-skin image detection
- **Streamlit** web app for easy image upload and real-time analysis
- Fitzpatrick skin type integration for personalized risk assessment
- Image quality assessment with user feedback
- Detailed clinical information per diagnosis

---

## 🩺 Supported Skin Conditions

| Code   | Condition                              | Type          |
|--------|----------------------------------------|---------------|
| AKIEC  | Actinic Keratosis / Bowen's Disease    | Pre-cancerous |
| BCC    | Basal Cell Carcinoma                   | Cancerous     |
| BKL    | Benign Keratosis                       | Benign        |
| DF     | Dermatofibroma                         | Benign        |
| MEL    | Melanoma                               | Cancerous     |
| NV     | Nevus / Mole                           | Benign        |
| VASC   | Vascular Lesion                        | Benign        |

---

## 📁 Project Structure

```
Skin-Cancer-Detection/
├── archive/
│   ├── HAM10000_metadata.csv
│   ├── hmnist_28_28_RGB.csv
│   ├── hmnist_28_28_L.csv
│   ├── hmnist_8_8_RGB.csv
│   ├── hmnist_8_8_L.csv
│   ├── sorted_by_dx/              ← dataset images organized by class
│   │   ├── akiec/
│   │   ├── bcc/
│   │   ├── bkl/
│   │   ├── df/
│   │   ├── mel/
│   │   ├── nv/
│   │   └── vasc/
│   └── skin_cancer_detector/
│       ├── app.py                 ← Streamlit web application
│       ├── train.py               ← Model training script
│       ├── infer.py               ← Inference / prediction script
│       ├── requirements.txt       ← Python dependencies
│       ├── model_artifacts/       ← Saved models (after training)
│       │   ├── cnn_model.pt
│       │   ├── xgb_model.json
│       │   ├── ood_isolation_forest.joblib
│       │   └── metadata.json
│       └── _quick_eval_bcc.py
├── main_accuracy_graph.py
└── README.md
```

---

## ✅ Prerequisites

- Python **3.8 or higher**
- pip (Python package manager)
- Git
- (Optional but recommended) GPU with CUDA for faster training

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Jaya-11/SKIN-CANCER-DETECTION.git
cd SKIN-CANCER-DETECTION
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r archive/skin_cancer_detector/requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch, xgboost, streamlit; print('✅ All dependencies installed successfully!')"
```

---

## 📦 Dataset Preparation

This project uses the **HAM10000** dataset.

### Option A — Download from Kaggle

1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Download and extract images into `archive/sorted_by_dx/` organized by class:

```
archive/sorted_by_dx/
├── akiec/   ← place akiec images here
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

3. Place `HAM10000_metadata.csv` in `archive/`

### Option B — Use Pre-processed CSV Files

If you already have the CSV files, place them in `archive/`:
- `hmnist_28_28_RGB.csv`
- `hmnist_28_28_L.csv`
- `hmnist_8_8_RGB.csv`
- `hmnist_8_8_L.csv`

---

## 🏋️ Train the Model

```bash
cd archive/skin_cancer_detector

python train.py --data_root ../../archive/sorted_by_dx --output_dir ./model_artifacts
```

**Available Training Arguments:**

| Argument        | Description                              | Default              |
|----------------|------------------------------------------|----------------------|
| `--data_root`  | Path to folder with class subdirectories | (required)           |
| `--output_dir` | Where to save trained models             | `./model_artifacts`  |
| `--batch_size` | Batch size for training                  | `32`                 |
| `--epochs`     | Number of training epochs                | `10`                 |
| `--lr`         | Learning rate                            | `1e-4`               |

**After training, `model_artifacts/` will contain:**
- `cnn_model.pt` — Trained ResNet-18
- `xgb_model.json` — Trained XGBoost classifier
- `ood_isolation_forest.joblib` — Out-of-distribution detector
- `metadata.json` — Model config and class mappings

---

## 🌐 Run the Web App

```bash
cd archive/skin_cancer_detector

streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

### How to Use the Web App

1. **Select Fitzpatrick Skin Type** from the sidebar
2. **Enter model directory path** (default: `./model_artifacts`)
3. **Upload a skin lesion image** (JPG, PNG)
4. Click **Analyze** — you'll get:
   - Predicted diagnosis with confidence score
   - Clinical information and risk level
   - Image quality feedback

---

## 🖥️ Command Line Inference

```bash
cd archive/skin_cancer_detector

python infer.py --model_dir ./model_artifacts --image_path path/to/your/image.jpg
```

### Using the Python API

```python
from archive.skin_cancer_detector.infer import SkinCancerDetector

# Load the detector
detector = SkinCancerDetector("./model_artifacts")

# Run prediction
result = detector.predict("path/to/skin/image.jpg")

if result["accepted"]:
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}")
else:
    print(f"Rejected   : {result['reason']}")
```

---

## 📈 Model Evaluation

### Quick BCC Evaluation

```bash
cd archive/skin_cancer_detector

python _quick_eval_bcc.py --model_dir ./model_artifacts --data_root ../../archive/sorted_by_dx
```

### Accuracy Graph

```bash
# From the project root
python main_accuracy_graph.py
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|--------|----------|
| `ModuleNotFoundError` | Run `pip install -r archive/skin_cancer_detector/requirements.txt` |
| `CUDA not available` | Training will fall back to CPU automatically |
| `Model artifacts not found` | Train the model first using `train.py` |
| Streamlit port conflict | Run `streamlit run app.py --server.port 8502` |
| Low confidence results | Ensure the image is a clear, well-lit skin lesion photo |

---

## ⚠️ Disclaimer

> This system is intended to **assist** healthcare professionals, not replace them.
> Do **not** use this tool as a substitute for professional medical diagnosis.
> Always consult a qualified dermatologist for accurate diagnosis and treatment.

---

## 📚 References

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. (2018). *The HAM10000 dataset*. Scientific Data.
- **ResNet**: He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- **XGBoost**: Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add YourFeature"`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.
