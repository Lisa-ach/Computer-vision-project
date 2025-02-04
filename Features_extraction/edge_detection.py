# Edge detection

# On va appliquer Canny, sobel, prewitt, ... à des images données en entrée
# En sortie, j'ai besoin d'un dataframe de features pour pouvoir évaluer avec les classifications

import cv2
import numpy as np
import pandas as pd
from skimage.filters import prewitt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_processing')))
import import_images as i

# 1. Extract Edge Detection features from all images
print("=========1. Extract Edge Detection features from all images=========")
features_list = []

for img in i.images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply edge detection methods
    edges_canny = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_edges = prewitt(gray)
    
    # Extract features
    features_vector = {
        'canny_edge_density': np.sum(edges_canny) / edges_canny.size,
        'sobelx_mean': np.mean(sobelx),
        'sobelx_std': np.std(sobelx),
        'sobely_mean': np.mean(sobely),
        'sobely_std': np.std(sobely),
        'prewitt_edge_density': np.sum(prewitt_edges) / prewitt_edges.size
    }
    
    features_list.append(features_vector)

# 2. Convert feature list to DataFrame
print("=========2. Convert feature list to DataFrame=========")
features_df = pd.DataFrame(features_list)

# 3. Normalize features (optional, can be tuned)
print("=========3. Normalize features=========")
features_df = (features_df - features_df.min()) / (features_df.max() - features_df.min())

print(features_df.head())
