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

# Step 1: Extract Edge Detection features from all images
print("=========1. Extract Edge Detection features from all images=========")
features_list = []

for img, label in zip(i.images, i.Y):  # Associating images with their labels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better edge detection
    
    # Apply different edge detection methods
    otsu_thresh = threshold_otsu(gray)
    edges_canny = cv2.Canny(gray, otsu_thresh * 0.5, otsu_thresh * 1.5)  # Canny edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
    prewitt_edges = prewitt(gray)  # Prewitt operator for edge detection
    laplacian_edges = laplace(gray)  # Laplacian filter for detecting high-frequency regions
    roberts_edges = roberts(gray)  # Roberts filter for diagonal edge detection
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')  # Local Binary Pattern (LBP) for texture analysis
    
    # Extract meaningful statistical features from edge-detected images
    features_list.append([
        np.sum(edges_canny) / edges_canny.size,
        np.mean(sobelx), np.std(sobelx),
        np.mean(sobely), np.std(sobely),
        np.sum(prewitt_edges) / prewitt_edges.size,
        np.sum(laplacian_edges) / laplacian_edges.size,
        np.sum(roberts_edges) / roberts_edges.size,
        np.mean(lbp), np.std(lbp)
    ])


# Step 2: Convert feature list to a DataFrame
print("=========2. Convert feature list to DataFrame=========")
features_df = pd.DataFrame(features_list, columns=[
    'canny_edge_density', 'sobelx_mean', 'sobelx_std',
    'sobely_mean', 'sobely_std', 'prewitt_edge_density',
    'laplacian_edge_density', 'roberts_edge_density',
    'lbp_mean', 'lbp_std'
])


# Step 3: Normalize features to ensure uniform scale for classification
print("=========3. Normalize features=========")
features_df = (features_df - features_df.min()) / (features_df.max() - features_df.min())

# Display the first few rows of the DataFrame to verify extracted features
print(features_df.head())