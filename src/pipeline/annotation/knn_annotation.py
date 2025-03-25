"""
Functions to help with annotation of functions using KNN-based clustering

This module provides an alternative approach to tissue segmentation using
K-Nearest Neighbors clustering of pixel values.

Steps:
1. Load image
2. Apply KNN clustering to segment tissues
3. Output image
4. Image is manually annotated
5. Image is loaded back in
6. Image is convex hull'd and regions segmented
7. Separately, user is asked to input tissue type
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
import slideio
import pylab as plt
from glob import glob
from tqdm import tqdm
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from skimage.morphology import binary_dilation as dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters.rank import gradient
from scipy.ndimage import binary_fill_holes

# Get auto_slide_dir from the initial_annotation.py file
auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

sys.path.append(os.path.join(auto_slide_dir, 'src'))
import utils

############################################################
# PARAMS
down_sample = 100
dilation_kern_size = 2
area_threshold = 10000
n_clusters = 5  # Number of clusters for KMeans initialization
n_neighbors = 7  # Number of neighbors for KNN
############################################################

def get_knn_mask(scene, down_sample=100, n_clusters=5, n_neighbors=7):
    """
    Get KNN-based segmentation mask for scene.
    
    Args:
        scene: slideio scene object
        down_sample: down sample factor
        n_clusters: number of clusters for KMeans initialization
        n_neighbors: number of neighbors for KNN
        
    Returns:
        bin_image: binary mask of segmented tissue
        labels_image: labeled image with cluster assignments
    """
    image_shape = scene.rect[2:]
    down_shape = [int(i/down_sample) for i in image_shape]
    small_image = scene.read_block(size=down_shape)
    if small_image.shape[:2] != down_shape:
        small_image = np.swapaxes(small_image, 0, 1)
    
    # Reshape image for clustering
    pixels = small_image.reshape(-1, 3)
    
    # Use KMeans to initialize labels (faster than KNN on all pixels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(pixels)
    
    # Train KNN on a subset of the data with KMeans labels
    sample_indices = np.random.choice(len(pixels), size=min(100000, len(pixels)), replace=False)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(pixels[sample_indices], kmeans_labels[sample_indices])
    
    # Predict on all pixels
    labels = knn.predict(pixels)
    
    # Reshape back to image dimensions
    labels_image = labels.reshape(small_image.shape[:2])
    
    # Create binary mask (assuming background is the most common label)
    background_label = np.bincount(labels).argmax()
    bin_image = labels_image != background_label
    
    return bin_image, labels_image

def process_slide(data_path, annot_dir):
    """Process a single slide with KNN-based segmentation"""
    file_basename = os.path.basename(data_path)
    slide = slideio.open_slide(data_path, 'SVS')
    scene = slide.get_scene(0)
    image_rect = np.array(scene.rect) // down_sample
    image = scene.read_block(size=image_rect[2:])

    ############################################################
    # Get KNN-based segmentation mask
    threshold_mask, cluster_labels = get_knn_mask(
        scene, 
        down_sample=down_sample,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors
    )

    dilation_kern = np.ones((dilation_kern_size, dilation_kern_size))
    dilated_mask = dilation(threshold_mask, footprint=dilation_kern)

    label_image = label(dilated_mask)
    regions = regionprops(label_image)
    image_label_overlay = label2rgb(label_image, image=dilated_mask, bg_label=0)

    wanted_feature_names = [
            'label',
            'area',
            'eccentricity',
            'axis_major_length',
            'axis_minor_length',
            'eccentricity',
            'solidity',
            'centroid',
            ]

    wanted_features = [
            [getattr(region, feature_name) for region in regions]
            for feature_name in wanted_feature_names
            ]

    region_frame = pd.DataFrame(
            {feature_name: feature for feature_name, feature in \
                    zip(wanted_feature_names, wanted_features)}
            )

    region_frame.sort_values('label', ascending=True, inplace=True)
    wanted_regions_frame = region_frame[region_frame.area > area_threshold]
    wanted_regions = wanted_regions_frame.label.values

    # Output wanted_regions_frame to be annotated manually
    wanted_regions_frame['tissue_type'] = np.nan
    wanted_regions_frame['tissue_num'] = np.nan

    # Save with _knn suffix to distinguish from image processing method
    wanted_regions_frame.to_csv(
            os.path.join(annot_dir, file_basename.replace('.svs', '_knn.csv')),
            index=False
            )

    # Drop regions that are not wanted
    fin_label_image = label_image.copy()
    for i in region_frame.label.values:
        if i not in wanted_regions:
            fin_label_image[fin_label_image == i] = 0

    # Write out image with regions labelled
    np.save(
            os.path.join(annot_dir, file_basename.replace('.svs', '_knn.npy')),
            fin_label_image
            )

    image_label_overlay = label2rgb(fin_label_image,
                                    image=fin_label_image>0,
                                    bg_label=0)

    filled_binary = binary_fill_holes(fin_label_image > 0)*1

    gradient_image = gradient(filled_binary, np.ones((10, 10)))
    grad_inds = np.where(gradient_image > 0)

    # Create visualization with both original and KNN-based segmentation
    fig, ax = plt.subplots(1, 6,
                          sharex=True, sharey=True,
                          figsize=(24, 10)
                          )
    ax[0].imshow(np.swapaxes(image, 0, 1))
    ax[1].imshow(threshold_mask)
    ax[2].imshow(dilated_mask)
    ax[3].imshow(image_label_overlay)
    ax[4].imshow(np.swapaxes(image, 0, 1))
    ax[4].scatter(grad_inds[1], grad_inds[0],
                  s=1, color='orange', alpha=0.7,
                  label='Outline')
    
    # Show cluster labels with a different colormap
    ax[5].imshow(cluster_labels, cmap='tab10')
    
    ax[0].set_title('Original')
    ax[1].set_title('KNN Threshold')
    ax[2].set_title('Dilated')
    ax[3].set_title('Labelled')
    ax[4].set_title('Outline Overlay')
    ax[5].set_title('KNN Clusters')
    
    ax[4].legend()
    
    # Add labels at the center of each region
    for this_row in wanted_regions_frame.itertuples():
        ax[3].text(this_row.centroid[1], this_row.centroid[0],
                   this_row.label, color='r', fontsize=25,
                   weight='bold')
    
    fig.suptitle(f"{file_basename} (KNN Method)")
    plt.tight_layout()
    fig.savefig(os.path.join(annot_dir, file_basename.replace('.svs', '_knn.png')),
                bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to run KNN-based annotation"""
    data_dir = os.path.join(auto_slide_dir, 'data')
    glob_pattern = 'TRI*.svs'
    file_list = glob(os.path.join(data_dir, glob_pattern))

    # Create a KNN-specific annotation directory
    annot_dir = os.path.join(data_dir, 'initial_annotation')
    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)

    for data_path in tqdm(file_list):
        process_slide(data_path, annot_dir)

if __name__ == "__main__":
    main()
