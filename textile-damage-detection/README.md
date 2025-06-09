# Textile Damage Detection

A computer vision-based system for detecting defects in textile materials using YOLOv8.

## Overview

This project implements a real-time defect detection system for textile manufacturing using YOLOv8. It can identify various types of fabric defects and damages, helping in quality control and reducing waste.

## Features

- Real-time defect detection
- High accuracy and speed
- Multiple defect type classification
- Easy integration with existing systems
- GPU acceleration support

## Dataset

The dataset is sourced from Roboflow and contains:
- Training images: `train/images/`
- Validation images: `valid/images/`
- Test images: `test/images/`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the YOLOv8 pretrained weights:
```bash
# The script will automatically download yolov8n.pt if not present
```

## Usage

### Training

To train the model:
```bash
python train.py
```

Training parameters can be modified in `train.py`:
- Batch size
- Image size
- Number of epochs
- Augmentation settings

### Inference

To run inference on new images:
```bash
python predict.py --source path/to/image.jpg
```

## Model Architecture

- Base model: YOLOv8-nano
- Input size: 640x640
- Number of classes: 1 (defect)
- Augmentation: Mosaic, Random flip

## Performance

The model achieves:
- High detection accuracy
- Real-time processing speed
- Low false positive rate

## Directory Structure

```
textile-damage-detection/
├── train/              # Training images and labels
├── valid/              # Validation images and labels
├── test/               # Test images and labels
├── runs/               # Training outputs and logs
├── train.py           # Training script
├── predict.py         # Inference script
├── data.yaml          # Dataset configuration
└── requirements.txt   # Project dependencies
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. 