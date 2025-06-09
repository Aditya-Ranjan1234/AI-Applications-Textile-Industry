# Garments Workers AI & ML Demo

An AI-powered assistance system for garment workers to improve quality control and efficiency.

## Overview

This project implements an AI system that helps garment workers with quality control, provides real-time feedback, and improves overall production efficiency. It uses computer vision and machine learning to assist workers in maintaining high-quality standards.

## Features

- Real-time quality control assistance
- Worker performance monitoring
- Training and guidance system
- Quality metrics tracking
- Error detection and prevention

## Components

### 1. Quality Control Module
- Defect detection
- Stitch quality assessment
- Pattern matching
- Color consistency check

### 2. Worker Assistance
- Real-time feedback
- Step-by-step guidance
- Error prevention alerts
- Performance tracking

### 3. Training System
- Interactive tutorials
- Skill assessment
- Progress tracking
- Performance analytics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up the camera system:
```bash
# Configure camera settings in config.yaml
```

## Usage

### Starting the System

To start the worker assistance system:
```bash
python main.py
```

### Training Mode

To access the training interface:
```bash
python training.py
```

### Monitoring Dashboard

To view the monitoring dashboard:
```bash
python dashboard.py
```

## System Architecture

- Frontend: Web-based interface
- Backend: Python Flask server
- Computer Vision: OpenCV, YOLOv8
- Database: SQLite/PostgreSQL
- Real-time processing pipeline

## Directory Structure

```
Garments Workers AI&ML demo/
├── src/                # Source code
│   ├── cv/           # Computer vision modules
│   ├── ml/           # Machine learning models
│   ├── web/          # Web interface
│   └── utils/        # Utility functions
├── data/             # Training data
├── models/           # Trained models
├── config/           # Configuration files
├── static/           # Static assets
├── templates/        # HTML templates
├── main.py          # Main application
├── training.py      # Training interface
├── dashboard.py     # Monitoring dashboard
└── requirements.txt # Project dependencies
```

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Flask
- SQLite/PostgreSQL
- Webcam/Camera system

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. 