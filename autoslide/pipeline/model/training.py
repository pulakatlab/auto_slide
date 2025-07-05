"""
Resources
- https://github.com/DatumLearning/Mask-RCNN-finetuning-PyTorch/blob/main/notebook.ipynb
- https://www.youtube.com/watch?v=vV9L71hK-RE
- https://www.youtube.com/watch?v=t1MrzuAUdoE

"""

#!/usr/bin/env python
# coding: utf-8


from autoslide.pipeline.model.training_utils import (
    setup_directories, setup_training, train_model, plot_losses, evaluate_model, load_model
)
from autoslide.pipeline.model.prediction_utils import initialize_model
from autoslide.pipeline.model.data_preprocessing import (
    prepare_data, plot_augmented_samples, create_sample_plots
)
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cv2
import torch
import argparse

# Import config
from autoslide import config

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = config['artifacts_dir']
plot_dir = config['plot_dirs']

# Import utilities directly

##############################
##############################


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Mask R-CNN model for vessel detection')
    parser.add_argument('--retrain', action='store_true', 
                       help='Force retraining even if a saved model exists')
    return parser.parse_args()


def main():
    """Main function to run the training pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    plot_dir, artifacts_dir = setup_directories(data_dir)

    # Prepare all data using the preprocessing pipeline
    data_components = prepare_data(data_dir, use_augmentation=True)
    
    # Extract components
    train_dl = data_components['train_dl']
    val_dl = data_components['val_dl']
    train_imgs = data_components['train_imgs']
    train_masks = data_components['train_masks']
    val_imgs = data_components['val_imgs']
    val_masks = data_components['val_masks']
    img_dir = data_components['img_dir']
    mask_dir = data_components['mask_dir']
    aug_img_dir = data_components['aug_img_dir']
    aug_mask_dir = data_components['aug_mask_dir']
    aug_img_names = data_components['aug_img_names']
    aug_mask_names = data_components['aug_mask_names']

    # Create visualization plots
    if len(aug_img_names) > 0:
        plot_augmented_samples(aug_img_dir, aug_mask_dir,
                               aug_img_names, aug_mask_names, plot_dir)

    create_sample_plots(
        train_imgs, train_masks, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        plot_dir
    )

    # Initialize model
    model = initialize_model()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Setup training
    optimizer = setup_training(model, device)

    # Model paths
    best_model_path = os.path.join(
        artifacts_dir, 'best_val_mask_rcnn_model.pth')

    # Load existing model or train new one
    if os.path.exists(best_model_path) and not args.retrain:
        print('Loading model from savefile')
        model = load_model(model, best_model_path, device)
    else:
        # Train model
        model, all_train_losses, all_val_losses, best_val_loss = train_model(
            model, train_dl, val_dl, optimizer, device, plot_dir, artifacts_dir
        )

    # Evaluate model
    evaluate_model(
        model, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        device, plot_dir
    )


if __name__ == "__main__":
    main()
