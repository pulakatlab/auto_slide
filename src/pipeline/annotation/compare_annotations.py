"""
Compare annotation results from different methods

This module provides functionality to compare the results of different
annotation methods (image processing vs. KNN-based) and generate
visualizations to help users select the best method.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.color import label2rgb

# Get auto_slide_dir from the initial_annotation.py file
auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

sys.path.append(os.path.join(auto_slide_dir, 'src'))
import utils

def compare_annotations(file_basename, annot_dir, output_dir):
    """
    Compare annotation results for a single file
    
    Args:
        file_basename: Base name of the file (without extension)
        annot_dir: Directory containing annotation files
        output_dir: Directory to save comparison visualizations
    """
    # Load standard annotation
    std_npy_path = os.path.join(annot_dir, f"{file_basename}.npy")
    std_csv_path = os.path.join(annot_dir, f"{file_basename}.csv")
    
    # Load KNN annotation
    knn_npy_path = os.path.join(annot_dir, f"{file_basename}_knn.npy")
    knn_csv_path = os.path.join(annot_dir, f"{file_basename}_knn.csv")
    
    # Check if both methods have results
    if not (os.path.exists(std_npy_path) and os.path.exists(knn_npy_path)):
        print(f"Missing annotation files for {file_basename}")
        return
    
    # Load label images
    std_label_image = np.load(std_npy_path)
    knn_label_image = np.load(knn_npy_path)
    
    # Load region data
    std_regions = pd.read_csv(std_csv_path)
    knn_regions = pd.read_csv(knn_csv_path)
    
    # Create comparison visualization
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Standard method
    std_overlay = label2rgb(std_label_image, image=std_label_image>0, bg_label=0)
    ax[0].imshow(std_overlay)
    ax[0].set_title(f"Standard Method\n({len(std_regions)} regions)")
    
    # KNN method
    knn_overlay = label2rgb(knn_label_image, image=knn_label_image>0, bg_label=0)
    ax[1].imshow(knn_overlay)
    ax[1].set_title(f"KNN Method\n({len(knn_regions)} regions)")
    
    # Difference visualization
    # Create a difference mask (areas where methods disagree)
    std_binary = std_label_image > 0
    knn_binary = knn_label_image > 0
    difference = np.logical_xor(std_binary, knn_binary)
    
    # Overlay the difference on the third plot
    ax[2].imshow(std_overlay, alpha=0.5)
    ax[2].imshow(knn_overlay, alpha=0.5)
    ax[2].imshow(np.ma.masked_where(~difference, difference), 
                 cmap='autumn', alpha=0.7)
    ax[2].set_title("Comparison\n(Red areas show differences)")
    
    # Add region counts and other metrics
    std_area = np.sum(std_binary)
    knn_area = np.sum(knn_binary)
    diff_area = np.sum(difference)
    agreement = 1 - (diff_area / max(std_area, knn_area))
    
    fig.suptitle(f"Annotation Comparison for {file_basename}\n" +
                 f"Agreement: {agreement:.2%}", fontsize=16)
    
    plt.tight_layout()
    
    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{file_basename}_comparison.png"),
                bbox_inches='tight')
    plt.close(fig)
    
    # Return metrics for summary
    return {
        'file': file_basename,
        'std_regions': len(std_regions),
        'knn_regions': len(knn_regions),
        'std_area': std_area,
        'knn_area': knn_area,
        'diff_area': diff_area,
        'agreement': agreement
    }

def main():
    """Main function to compare annotation methods"""
    data_dir = os.path.join(auto_slide_dir, 'data')
    annot_dir = os.path.join(data_dir, 'initial_annotation')
    comparison_dir = os.path.join(data_dir, 'annotation_comparison')
    
    # Get all standard annotation files
    std_npy_files = glob(os.path.join(annot_dir, '*.npy'))
    std_basenames = [os.path.basename(f).replace('.npy', '') for f in std_npy_files 
                    if not f.endswith('_knn.npy')]
    
    # Compare annotations for each file
    comparison_metrics = []
    for basename in tqdm(std_basenames):
        metrics = compare_annotations(basename, annot_dir, comparison_dir)
        if metrics:
            comparison_metrics.append(metrics)
    
    # Create summary report
    if comparison_metrics:
        metrics_df = pd.DataFrame(comparison_metrics)
        
        # Calculate overall statistics
        avg_agreement = metrics_df['agreement'].mean()
        std_better_count = sum(metrics_df['std_regions'] > metrics_df['knn_regions'])
        knn_better_count = sum(metrics_df['knn_regions'] > metrics_df['std_regions'])
        equal_count = sum(metrics_df['std_regions'] == metrics_df['knn_regions'])
        
        # Create summary visualization
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Agreement histogram
        ax[0].hist(metrics_df['agreement'], bins=20, alpha=0.7)
        ax[0].axvline(avg_agreement, color='r', linestyle='--', 
                     label=f'Avg: {avg_agreement:.2%}')
        ax[0].set_title('Distribution of Agreement Between Methods')
        ax[0].set_xlabel('Agreement')
        ax[0].set_ylabel('Count')
        ax[0].legend()
        
        # Region count comparison
        ax[1].scatter(metrics_df['std_regions'], metrics_df['knn_regions'], alpha=0.7)
        ax[1].plot([0, max(metrics_df['std_regions'].max(), metrics_df['knn_regions'].max())], 
                  [0, max(metrics_df['std_regions'].max(), metrics_df['knn_regions'].max())], 
                  'k--')
        ax[1].set_title('Region Count Comparison')
        ax[1].set_xlabel('Standard Method Regions')
        ax[1].set_ylabel('KNN Method Regions')
        
        fig.suptitle('Annotation Method Comparison Summary', fontsize=16)
        plt.tight_layout()
        
        # Save summary
        fig.savefig(os.path.join(comparison_dir, 'summary_comparison.png'),
                   bbox_inches='tight')
        
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(comparison_dir, 'comparison_metrics.csv'), index=False)
        
        # Print summary
        print(f"\nAnnotation Comparison Summary:")
        print(f"Total files compared: {len(comparison_metrics)}")
        print(f"Average agreement: {avg_agreement:.2%}")
        print(f"Files where standard method found more regions: {std_better_count}")
        print(f"Files where KNN method found more regions: {knn_better_count}")
        print(f"Files with equal region counts: {equal_count}")
        print(f"\nDetailed results saved to {comparison_dir}")

if __name__ == "__main__":
    main()
