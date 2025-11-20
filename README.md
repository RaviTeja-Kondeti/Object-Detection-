# Multi-Task Object Detection with VGG16

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📋 Overview

A deep learning implementation of multi-task object detection combining **classification** and **bounding box localization** using transfer learning with VGG16 backbone. The model simultaneously predicts object classes and their spatial locations in images from the Caltech-101 dataset.

## ✨ Key Features

- **Transfer Learning**: Leverages pre-trained VGG16 (ImageNet weights) as feature extractor
- **Multi-Task Architecture**: Dual-branch network for classification and regression
- **Multi-Loss Optimization**: Combines categorical cross-entropy and mean squared error
- **Caltech-101 Dataset**: Trained on 3 object classes (Airplanes, Faces, Motorbikes)
- **TensorFlow/Keras Functional API**: Flexible multi-output model architecture

## 🏗️ Architecture

```
Input Image (224×224×3)
        ↓
   VGG16 Backbone
   (frozen weights)
        ↓
    Flatten
        ↓
        ├──────────────────┐
        ↓                  ↓
Classification Branch  Regression Branch
(Dense: 512→512→3)    (Dense: 128→64→32→4)
  (softmax)            (sigmoid)
        ↓                  ↓
   Class Label      Bounding Box
```

## 🛠️ Technical Stack

- **Framework**: TensorFlow 2.x / Keras
- **Architecture**: VGG16 (transfer learning)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Functions**: 
  - Classification: Categorical Cross-Entropy
  - Regression: Mean Squared Error
- **Metrics**: Accuracy (classification), MSE (localization)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RaviTeja-Kondeti/Object-Detection-.git
cd Object-Detection-

# Install required packages
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib
```

## 🚀 Usage

### Training the Model

```python
# Load the Jupyter notebook
jupyter notebook 23302_RaviTejaKondetiAssignment3_github.ipynb

# Or run directly in Google Colab
# The notebook includes:
# 1. Data loading and preprocessing
# 2. Model architecture definition
# 3. Training with multi-task loss
# 4. Evaluation and visualization
```

### Dataset Structure

The Caltech-101 dataset should be organized as:
```
caltech-101/
├── images/
│   ├── airplanes/
│   ├── Faces/
│   └── Motorbikes/
└── gt_summary.csv  # Ground truth annotations
```

## 📊 Results

The model achieves multi-task learning by optimizing both:
- **Classification Loss**: Predicts object category
- **Regression Loss**: Predicts bounding box coordinates (xmin, ymin, xmax, ymax)

## 🎯 Model Details

### Data Split
- **Training**: 75%
- **Validation**: 15%
- **Test**: 10%

### Preprocessing
- Image resizing: 224×224 pixels
- Normalization: Pixel values scaled to [0, 1]
- Bounding box normalization: Coordinates scaled to image dimensions

### Training Configuration
- **Loss Weights**: Classification: 1.0, Regression: 1.0
- **Batch Processing**: Custom data loading pipeline
- **Image Augmentation**: Supported via TensorFlow preprocessing

## 📁 Project Structure

```
.
├── 23302_RaviTejaKondetiAssignment3_github.ipynb  # Main implementation
├── README.md                                       # This file
└── LICENSE                                         # MIT License
```

## 🔬 Key Implementation Highlights

1. **VGG16 Backbone**: Frozen convolutional layers for efficient feature extraction
2. **Dual Branch Design**: Separate dense layers for classification and localization tasks
3. **Custom Training Loop**: Handles multiple outputs and loss functions
4. **Functional API**: Flexible model architecture with multiple inputs/outputs

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/RaviTeja-Kondeti/Object-Detection-/issues).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ravi Teja Kondeti**
- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)

## 🙏 Acknowledgments

- VGG16 architecture from [Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- Caltech-101 dataset from [Fei-Fei et al., 2007](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- TensorFlow/Keras documentation and community

## 📚 References

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Keras Functional API](https://www.tensorflow.org/guide/keras/functional)

---

⭐ If you find this project useful, please consider giving it a star!
