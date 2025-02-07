import cv2
import numpy as np
import pandas as pd
from skimage.filters import prewitt, roberts, laplace, threshold_otsu, scharr
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
        # Check if it is not already in grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        else:
            gray = img
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Apply different edge detection methods
        otsu_thresh = threshold_otsu(gray)
        edges_canny = cv2.Canny(gray, otsu_thresh * 0.5, otsu_thresh * 1.5)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        prewitt_edges = prewitt(gray)
        laplacian_edges = laplace(gray)
        roberts_edges = roberts(gray)
        scharrx = scharr(gray, axis=0)
        scharry = scharr(gray, axis=1)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Extract meaningful statistical features
        features_list.append([
            np.mean(edges_canny), np.std(edges_canny),
            np.mean(sobelx), np.std(sobelx),
            np.mean(sobely), np.std(sobely),
            np.mean(prewitt_edges), np.std(prewitt_edges),
            np.mean(laplacian_edges), np.std(laplacian_edges),
            np.mean(roberts_edges), np.std(roberts_edges),
            np.mean(scharrx), np.std(scharrx),
            np.mean(scharry), np.std(scharry),
            np.mean(adaptive_thresh), np.std(adaptive_thresh),
            np.mean(lbp), np.std(lbp)
        ])

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list, columns=[
        'canny_mean', 'canny_std',
        'sobelx_mean', 'sobelx_std',
        'sobely_mean', 'sobely_std',
        'prewitt_mean', 'prewitt_std',
        'laplacian_mean', 'laplacian_std',
        'roberts_mean', 'roberts_std',
        'scharrx_mean', 'scharrx_std',
        'scharry_mean', 'scharry_std',
        'adaptive_thresh_mean', 'adaptive_thresh_std',
        'lbp_mean', 'lbp_std'
    ])

    # Normalize features
    features_df = (features_df - features_df.min()) / (features_df.max() - features_df.min())
    
    return features_df

# Extraction des features initiales
print("========= Extracting Edge Detection Features from Original Images =========")
features_df = extract_features_from_images(i.images)
