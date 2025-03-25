"""
Select preferred annotation method

This script helps users select their preferred annotation method
(image processing or KNN-based) and prepares the selected annotations
for the next steps in the pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import shutil
from tqdm import tqdm

# Get auto_slide_dir from the initial_annotation.py file
auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

sys.path.append(os.path.join(auto_slide_dir, 'src'))
import utils

def select_method(method='image_processing'):
    """
    Select the preferred annotation method and prepare files
    
    Args:
        method: The preferred method ('image_processing' or 'knn')
    """
    data_dir = os.path.join(auto_slide_dir, 'data')
    annot_dir = os.path.join(data_dir, 'initial_annotation')
    selected_dir = os.path.join(data_dir, 'selected_annotation')
    
    # Create selected annotation directory
    os.makedirs(selected_dir, exist_ok=True)
    
    # Get all annotation files
    if method == 'image_processing':
        npy_files = glob(os.path.join(annot_dir, '*.npy'))
        csv_files = glob(os.path.join(annot_dir, '*.csv'))
        png_files = glob(os.path.join(annot_dir, '*.png'))
        
        # Filter out KNN files
        npy_files = [f for f in npy_files if not f.endswith('_knn.npy')]
        csv_files = [f for f in csv_files if not f.endswith('_knn.csv')]
        png_files = [f for f in png_files if not f.endswith('_knn.png')]
    else:  # method == 'knn'
        npy_files = glob(os.path.join(annot_dir, '*_knn.npy'))
        csv_files = glob(os.path.join(annot_dir, '*_knn.csv'))
        png_files = glob(os.path.join(annot_dir, '*_knn.png'))
    
    # Copy files to selected directory
    print(f"Selecting {method} annotation files...")
    
    for src_file in tqdm(npy_files + csv_files + png_files):
        filename = os.path.basename(src_file)
        
        # For KNN files, remove the _knn suffix
        if method == 'knn':
            filename = filename.replace('_knn', '')
        
        dst_file = os.path.join(selected_dir, filename)
        shutil.copy2(src_file, dst_file)
    
    print(f"Selected {len(npy_files)} annotation files using {method} method")
    print(f"Files copied to {selected_dir}")
    
    # Create a selection record
    with open(os.path.join(selected_dir, 'selection_info.txt'), 'w') as f:
        f.write(f"Selected annotation method: {method}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of files: {len(npy_files)}\n")
    
    return True

def main():
    """Main function to select annotation method"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select preferred annotation method")
    parser.add_argument('--method', type=str, default='image_processing',
                        choices=['image_processing', 'knn'],
                        help='Preferred annotation method')
    args = parser.parse_args()
    
    select_method(args.method)

if __name__ == "__main__":
    main()
