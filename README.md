# Skin Cancer Detection System

An advanced AI-powered skin cancer detection system that combines deep learning feature extraction with machine learning classification for accurate and reliable diagnosis. This project uses ResNet-18 for image feature extraction, XGBoost for classification, and Isolation Forest for out-of-distribution detection, providing a complete pipeline for skin lesion analysis.

## 🚀 Features

- **Deep Learning Feature Extraction**: Utilizes pre-trained ResNet-18 model for robust image feature extraction
- **Machine Learning Classification**: XGBoost classifier for accurate skin cancer type prediction
- **Out-of-Distribution Detection**: Isolation Forest to identify non-skin images or unusual cases
- **Web Interface**: Streamlit-based web application for easy image upload and analysis
- **Clinical Risk Assessment**: Provides detailed clinical information and risk assessment for each diagnosis
- **Fitzpatrick Skin Type Integration**: Considers skin type for personalized risk assessment
- **Image Quality Assessment**: Evaluates image quality and provides feedback for better results

## 📊 Supported Skin Conditions

The system can detect the following skin conditions:

- **AKIEC**: Actinic Keratosis / Bowen's Disease (Pre-cancerous)
- **BCC**: Basal Cell Carcinoma (Most common skin cancer)
- **BKL**: Benign Keratosis (Harmless skin growth)
- **DF**: Dermatofibroma (Benign skin bump)
- **MEL**: Melanoma (Most dangerous skin cancer)
- **NV**: Nevus/Mole (Common skin growth)
- **VASC**: Vascular Lesion (Blood vessel abnormalities)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Skin-Cancer-Detection.git
   cd Skin-Cancer-Detection
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r archive/skin_cancer_detector/requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch, xgboost, streamlit; print('All dependencies installed successfully')"
   ```

## 📁 Dataset Preparation

The project uses the HAM10000 dataset. Follow these steps to prepare the data:

### Download HAM10000 Dataset

1. Download the HAM10000 dataset from [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2. Extract the images to `archive/sorted_by_dx/` directory with the following structure:
   ```
   archive/sorted_by_dx/
   ├── akiec/
   ├── bcc/
   ├── bkl/
   ├── df/
   ├── mel/
   ├── nv/
   └── vasc/
   ```

3. Place the metadata file `HAM10000_metadata.csv` in `archive/` directory

### Alternative: Use Pre-processed Data

If you have the processed CSV files, place them in `archive/`:
- `hmnist_28_28_L.csv` (grayscale 28x28)
- `hmnist_28_28_RGB.csv` (RGB 28x28)
- `hmnist_8_8_L.csv` (grayscale 8x8)
- `hmnist_8_8_RGB.csv` (RGB 8x8)

## 🏋️ Model Training

### Step 1: Prepare Training Data

Ensure your dataset is organized as described in the Dataset Preparation section.

### Step 2: Train the Model

Run the training script:

```bash
cd archive/skin_cancer_detector
python train.py --data_root ../../archive/sorted_by_dx --output_dir ./model_artifacts
```

**Training Parameters:**
- `--data_root`: Path to the directory containing class subdirectories with images
- `--output_dir`: Directory to save trained models and metadata (default: ./model_artifacts)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)

### Step 3: Monitor Training

The training script will:
- Load and preprocess images
- Train ResNet-18 for feature extraction
- Train XGBoost classifier on extracted features
- Train Isolation Forest for OOD detection
- Save model artifacts to the output directory

### Expected Output Files

After training, you'll find these files in `model_artifacts/`:
- `cnn_model.pt`: Trained ResNet-18 model
- `xgb_model.json`: Trained XGBoost classifier
- `ood_isolation_forest.joblib`: Trained Isolation Forest
- `metadata.json`: Model metadata and configuration

## 🔍 Inference and Prediction

### Using the Python API

```python
from archive.skin_cancer_detector.infer import SkinCancerDetector

# Initialize detector
detector = SkinCancerDetector("./model_artifacts")

# Make prediction
result = detector.predict("path/to/skin/image.jpg")

# Check result
if result["accepted"]:
    print(f"Predicted class: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
else:
    print(f"Rejected: {result['reason']}")
```

### Command Line Inference

```bash
cd archive/skin_cancer_detector
python infer.py --model_dir ./model_artifacts --image_path path/to/image.jpg
```

## 🌐 Web Application

### Running the Streamlit App

1. **Navigate to the app directory**
   ```bash
   cd archive/skin_cancer_detector
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

### Using the Web Interface

1. **Select Fitzpatrick Skin Type**: Choose your skin type from the sidebar for personalized assessment
2. **Specify Model Directory**: Enter the path to your model artifacts (default: ./model_artifacts)
3. **Upload Image**: Drag and drop or browse to select a skin lesion image
4. **Get Results**: The system will analyze the image and provide:
   - Predicted diagnosis
   - Confidence score
   - Clinical information
   - Risk assessment
   - Image quality feedback

## 📈 Model Evaluation

### Quick Evaluation Script

Use the provided evaluation script:

```bash
cd archive/skin_cancer_detector
python _quick_eval_bcc.py --model_dir ./model_artifacts --data_root ../../archive/sorted_by_dx
```

### Accuracy Graph

Run the main accuracy graph script:

```bash
python main_accuracy_graph.py
```

This will generate performance visualizations for the trained model.

## 🔧 Configuration and Customization

### Model Configuration

Edit `metadata.json` in the model artifacts directory to adjust:
- Skin ratio threshold for image acceptance
- Class names and mappings
- Model hyperparameters

### Web App Customization

Modify `app.py` to:
- Add new disease information
- Customize the user interface
- Integrate additional clinical features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This AI system is designed to assist healthcare professionals and should not be used as a substitute for professional medical diagnosis. Always consult with qualified medical personnel for accurate diagnosis and treatment.

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-username/Skin-Cancer-Detection/issues) page
2. Create a new issue with detailed description
3. Include error messages, system information, and steps to reproduce

## 🔗 References

- HAM10000 Dataset: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
- ResNet: He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. CVPR 2016.
- XGBoost: Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. KDD 2016.
