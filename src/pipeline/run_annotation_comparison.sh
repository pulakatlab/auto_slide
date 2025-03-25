#!/bin/bash

# Run annotation comparison script
# This script runs both annotation methods and generates comparison visualizations

echo "Running AutoSlide annotation comparison..."

# Run both annotation methods and generate comparisons
python run_pipeline.py --skip_training --annotation_method both --compare_annotations

echo "Annotation comparison complete. Please review the comparison visualizations in data/annotation_comparison/"
echo "Then run the pipeline with your preferred method using:"
echo "  python run_pipeline.py --select_method [image_processing|knn]"
