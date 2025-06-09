from ultralytics import YOLO
import os
import torch

def train_yolo_model():
    # Check for GPU availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model
    
    # Training parameters
    training_args = {
        'data': 'data.yaml',           # Path to data config file
        'epochs': 100,                 # Number of training epochs
        'imgsz': 640,                  # Image size
        'batch': 8,                    # Reduced batch size to prevent memory issues
        'patience': 50,                # Early stopping patience
        'device': device,              # Automatically use GPU if available
        'workers': 4,                  # Reduced number of worker threads
        'project': 'runs/train',       # Project name
        'name': 'textile_defect',      # Experiment name
        'exist_ok': True,              # Overwrite existing experiment
        'pretrained': True,            # Use pretrained weights
        'optimizer': 'auto',           # Optimizer (SGD, Adam, etc.)
        'verbose': True,               # Print verbose output
        'seed': 42,                    # Random seed for reproducibility
        'deterministic': True,         # Deterministic training
        'mosaic': 0.5,                 # Reduced mosaic augmentation probability
        'mixup': 0.0,                  # Disabled mixup augmentation
        'copy_paste': 0.0,             # Disabled copy-paste augmentation
        'cache': False,                # Disabled caching to save memory
        'amp': True,                   # Enable automatic mixed precision
    }
    
    # Start training
    results = model.train(**training_args)
    
    # Save the trained model
    model.save('textile_defect_model.pt')
    
    return results

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('runs/train', exist_ok=True)
    
    # Train the model
    results = train_yolo_model()
    
    # Print training results
    print("\nTraining completed!")
    print(f"Best mAP50: {results.best_map50}")
    print(f"Best epoch: {results.best_epoch}") 