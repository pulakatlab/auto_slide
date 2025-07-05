"""
Batch prediction script for vessel segmentation on all images in suggested_regions.

This script:
1. Finds all images under data_dir/suggested_regions/*/images/*.png
2. Checks if corresponding masks already exist
3. Performs prediction only on images without existing masks
4. Saves predicted masks to corresponding mask directories
"""

from autoslide.pipeline import utils
import os
import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from importlib import reload
from autoslide import config
from tqdm import tqdm
import argparse

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = config['artifacts_dir']
plot_dir = config['plot_dirs']


def load_model(model_path=None):
    """
    Load the trained Mask R-CNN model.
    
    Args:
        model_path (str): Path to the saved model. If None, uses default path.
        
    Returns:
        tuple: (model, device, transform) - Loaded model, device, and transform
    """
    if model_path is None:
        model_path = os.path.join(artifacts_dir, 'best_val_mask_rcnn_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Recreate base model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, 2)
    
    # Load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    transform = T.ToTensor()
    
    print(f"Model loaded from: {model_path}")
    print(f"Using device: {device}")
    
    return model, device, transform


def find_images_to_process():
    """
    Find all images that need prediction and don't already have masks.
    
    Returns:
        list: List of dictionaries containing image paths and corresponding mask paths
    """
    suggested_regions_dir = os.path.join(data_dir, 'suggested_regions')
    
    if not os.path.exists(suggested_regions_dir):
        print(f"Suggested regions directory not found: {suggested_regions_dir}")
        return []
    
    # Find all image files
    image_pattern = os.path.join(suggested_regions_dir, '**', 'images', '*.png')
    image_paths = glob(image_pattern, recursive=True)
    
    print(f"Found {len(image_paths)} images in suggested_regions")
    
    images_to_process = []
    
    for image_path in image_paths:
        # Determine corresponding mask path
        # Replace 'images' with 'masks' and add '_mask' suffix
        mask_path = image_path.replace('/images/', '/masks/')
        mask_path = mask_path.replace('.png', '_mask.png')
        
        # Check if mask already exists
        if not os.path.exists(mask_path):
            # Create mask directory if it doesn't exist
            mask_dir = os.path.dirname(mask_path)
            os.makedirs(mask_dir, exist_ok=True)
            
            images_to_process.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'image_name': os.path.basename(image_path)
            })
        else:
            print(f"Mask already exists for {os.path.basename(image_path)}, skipping")
    
    print(f"Found {len(images_to_process)} images that need prediction")
    return images_to_process


def predict_single_image(model, image_path, device, transform):
    """
    Perform prediction on a single image.
    
    Args:
        model: Trained Mask R-CNN model
        image_path (str): Path to the input image
        device: PyTorch device
        transform: Image transformation function
        
    Returns:
        numpy.ndarray: Combined predicted mask
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).to(device)
        
        # Perform prediction
        with torch.no_grad():
            pred = model([img_tensor])
        
        # Combine all predicted masks
        n_preds = len(pred[0]["masks"])
        if n_preds > 0:
            # Stack all predictions and take the mean
            all_preds = np.stack([
                (pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze()
                for i in range(n_preds)
            ])
            
            # Weight by confidence scores if available
            if len(pred[0]["scores"]) > 0:
                scores = pred[0]["scores"].cpu().detach().numpy()
                weighted_mask = np.zeros_like(all_preds[0], dtype=np.float32)
                total_weight = 0
                
                for mask, score in zip(all_preds, scores):
                    weighted_mask += mask.astype(np.float32) * score
                    total_weight += score
                
                if total_weight > 0:
                    combined_mask = (weighted_mask / total_weight).astype(np.uint8)
                else:
                    combined_mask = all_preds.mean(axis=0).astype(np.uint8)
            else:
                combined_mask = all_preds.mean(axis=0).astype(np.uint8)
        else:
            # No predictions found
            img_array = np.array(img)
            combined_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        return combined_mask
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def save_prediction_visualization(image_path, mask_path, predicted_mask, plot_dir):
    """
    Save a visualization of the prediction for quality control.
    
    Args:
        image_path (str): Path to original image
        mask_path (str): Path where mask will be saved
        predicted_mask (numpy.ndarray): Predicted mask
        plot_dir (str): Directory to save visualization
    """
    try:
        # Create visualization directory
        vis_dir = os.path.join(plot_dir, 'prediction_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load original image
        img = Image.open(image_path).convert("RGB")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = np.array(img)
        mask_binary = predicted_mask > 127
        overlay[mask_binary] = [255, 0, 0]  # Red overlay for vessels
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = Vessels)')
        axes[2].axis('off')
        
        # Save visualization
        vis_filename = os.path.basename(image_path).replace('.png', '_prediction.png')
        vis_path = os.path.join(vis_dir, vis_filename)
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating visualization for {image_path}: {e}")


def process_all_images(model_path=None, save_visualizations=False, max_images=None):
    """
    Process all images that need prediction.
    
    Args:
        model_path (str): Path to saved model
        save_visualizations (bool): Whether to save prediction visualizations
        max_images (int): Maximum number of images to process (for testing)
    """
    print("Starting batch prediction on suggested regions...")
    
    # Load model
    model, device, transform = load_model(model_path)
    
    # Find images to process
    images_to_process = find_images_to_process()
    
    if not images_to_process:
        print("No images found that need prediction.")
        return
    
    # Limit number of images if specified
    if max_images and max_images < len(images_to_process):
        images_to_process = images_to_process[:max_images]
        print(f"Limited processing to {max_images} images for testing")
    
    # Process each image
    successful_predictions = 0
    failed_predictions = 0
    
    for item in tqdm(images_to_process, desc="Processing images"):
        image_path = item['image_path']
        mask_path = item['mask_path']
        image_name = item['image_name']
        
        # Perform prediction
        predicted_mask = predict_single_image(model, image_path, device, transform)
        
        if predicted_mask is not None:
            try:
                # Save predicted mask
                mask_img = Image.fromarray(predicted_mask, mode='L')
                mask_img.save(mask_path)
                
                # Save visualization if requested
                if save_visualizations:
                    save_prediction_visualization(image_path, mask_path, predicted_mask, plot_dir)
                
                successful_predictions += 1
                
            except Exception as e:
                print(f"Error saving mask for {image_name}: {e}")
                failed_predictions += 1
        else:
            failed_predictions += 1
    
    print(f"\nBatch prediction complete!")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Total processed: {successful_predictions + failed_predictions}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch prediction on suggested regions')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model (default: best_val_mask_rcnn_model.pth)')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Save prediction visualizations for quality control')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    process_all_images(
        model_path=args.model_path,
        save_visualizations=args.save_visualizations,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
