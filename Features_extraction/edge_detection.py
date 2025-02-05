import cv2
import numpy as np
import pandas as pd
from skimage.filters import prewitt, roberts, laplace, threshold_otsu
from skimage.feature import local_binary_pattern
import sys
import os

# Add the data processing module to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_processing')))
import import_images as i

def extract_features_from_images(image_list):
    """
    Extract edge detection features from a given list of images.

    Args:
    image_list (list): List of images (NumPy arrays)

    Returns:
    pd.DataFrame: DataFrame containing extracted edge detection features.
    """
    features_list = []

    for img in image_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Apply different edge detection methods
        otsu_thresh = threshold_otsu(gray)
        edges_canny = cv2.Canny(gray, otsu_thresh * 0.5, otsu_thresh * 1.5)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        prewitt_edges = prewitt(gray)
        laplacian_edges = laplace(gray)
        roberts_edges = roberts(gray)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Extract meaningful statistical features
        features_list.append([
            np.sum(edges_canny) / edges_canny.size,
            np.mean(sobelx), np.std(sobelx),
            np.mean(sobely), np.std(sobely),
            np.sum(prewitt_edges) / prewitt_edges.size,
            np.sum(laplacian_edges) / laplacian_edges.size,
            np.sum(roberts_edges) / roberts_edges.size,
            np.mean(lbp), np.std(lbp)
        ])

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list, columns=[
        'canny_edge_density', 'sobelx_mean', 'sobelx_std',
        'sobely_mean', 'sobely_std', 'prewitt_edge_density',
        'laplacian_edge_density', 'roberts_edge_density',
        'lbp_mean', 'lbp_std'
    ])

    # Normalize features
    features_df = (features_df - features_df.min()) / (features_df.max() - features_df.min())
    
    return features_df

# Extraction des features initiales
print("========= Extracting Edge Detection Features from Original Images =========")
features_df = extract_features_from_images(i.images)
