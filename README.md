# AI Applications in Textile-Garment Industry

> A comprehensive collection of AI/ML solutions for textile manufacturing, including defect detection, demand forecasting, worker assistance systems, and fashion product classification.

This repository contains various AI/ML applications focused on solving challenges in the textile and garment industry. The projects demonstrate the practical implementation of artificial intelligence in different aspects of textile manufacturing and quality control.

## Projects Overview

### 1. Textile Damage Detection
**Location**: `textile-damage-detection/`
- YOLOv8-based defect detection system
- Real-time quality control for textile manufacturing
- Detects various types of fabric defects and damages
- Uses computer vision for automated quality inspection

### 2. Garment Worker AI & ML Demo
**Location**: `Garments Workers AI&ML demo/`
- AI-powered worker assistance system
- Helps garment workers with quality control
- Provides real-time feedback and guidance
- Improves worker efficiency and product quality

### 3. Textile Demand Forecasting
**Location**: `textile-demand-forecasting/`
- Machine learning-based demand prediction
- Helps in inventory management
- Optimizes production planning
- Reduces waste and improves resource allocation

### 4. Demand Prediction
**Location**: `demand prediction/`
- Advanced demand forecasting models
- Market trend analysis
- Seasonal pattern recognition
- Helps in strategic decision making

### 5. Precision Fashion Image Classification
**Location**: `Precision-fashion-image-classification/`
- HOG and KNN-based fashion product classification
- 98% classification accuracy
- Fast and efficient inference
- No deep learning required
- Works with limited computational resources

## Technology Stack

- **Computer Vision**: YOLOv8, OpenCV, HOG
- **Machine Learning**: PyTorch, scikit-learn, KNN
- **Deep Learning**: Neural Networks, Transfer Learning
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly

## Getting Started

Each project directory contains its own:
- `requirements.txt` for dependencies
- `README.md` with specific instructions
- Training scripts and models
- Dataset information

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for computer vision tasks)
- Sufficient RAM (16GB+ recommended)

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies for specific project:
```bash
cd [project-directory]
pip install -r requirements.txt
```

## Project Structure
```
├── textile-damage-detection/     # Defect detection system
├── Garments Workers AI&ML demo/  # Worker assistance system
├── textile-demand-forecasting/   # Demand prediction system
├── demand prediction/           # Market analysis system
├── Precision-fashion-image-classification/  # Fashion product classification
├── venv/                        # Virtual environment
└── README.md                    # This file
```

## Features

### Textile Damage Detection
- Real-time defect detection
- Multiple defect type classification
- High accuracy and speed
- Easy integration with existing systems

### Garment Worker AI
- Real-time assistance
- Quality control guidance
- Performance monitoring
- Training support

### Demand Forecasting
- Accurate demand prediction
- Market trend analysis
- Inventory optimization
- Resource planning

### Precision Fashion Classification
- High accuracy (98%)
- Fast inference
- Simple and interpretable model
- Works with limited resources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Roboflow for dataset support
- Ultralytics for YOLOv8 implementation
- Open source community for various tools and libraries

## Contact

For any queries or suggestions, please open an issue in the repository. 