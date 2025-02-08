import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters import prewitt, roberts, laplace, threshold_otsu, scharr
from skimage.feature import local_binary_pattern
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_processing')))
# import import_images as i

class FeatureExtraction:

    def __init__(self, images):
        
        self.images = images
        
    # ===================================================================================================
    # 1. Interest points detection methods
        
    def method_SIFT (self):
        print("============================================")
        print("\033[1mExtracting SIFT Features\033[0;0m")
        # 1. Extract SIFT features from all images
        sift = cv2.SIFT_create()
        descriptors_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
                # Display the image
                # plt.figure(figsize=(8, 6))
                # plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
                # plt.axis('off')
                # plt.title(f"SIFT Features ({len(keypoints)} Keypoints)")
                # plt.show()

        # 2. Stack all descriptors for clustering (BoVW)
        all_descriptors = np.vstack(descriptors_list)

        # 3. Cluster descriptors using KMeans (BoVW approach)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)

        # 4. Create feature histograms for each image
        def extract_features(image_descriptors, kmeans_model, num_clusters):
            feature_histogram = np.zeros(num_clusters)
            if image_descriptors is not None:
                labels = kmeans_model.predict(image_descriptors)
                for label in labels:
                    feature_histogram[label] += 1
            return feature_histogram

        feature_vectors = [extract_features(desc, kmeans, num_clusters) for desc in descriptors_list]

        return feature_vectors

    def method_ORB(self):
        print("============================================")
        print("\033[1mExtracting ORB Features\033[0;0m")
        # 1. Extract ORB features from all images
        orb = cv2.ORB_create()
        descriptors_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is None: # if not descriptor found, add by default a vector of zeros
                descriptors = np.zeros((0, 32), dtype=np.uint8)
           
            descriptors_list.append(descriptors)

        # 2. Stack all descriptors for clustering (BoVW)
        non_empty_descriptors = [desc for desc in descriptors_list if desc.shape[0] > 0] # remove zeros vectors
        all_descriptors = np.vstack(descriptors_list)

        # 3. Cluster descriptors using KMeans (BoVW approach)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)

        feature_vectors = []
        for desc in descriptors_list:
            if desc.shape[0] == 0:
                # Image without descriptors => null histogram
                feature_histogram = np.zeros(num_clusters, dtype=np.float32)
            else:
                labels = kmeans.predict(desc)
                feature_histogram = np.zeros(num_clusters, dtype=np.float32)
                for label in labels:
                    feature_histogram[label] += 1
            
            feature_vectors.append(feature_histogram)

        return feature_vectors
    
    # ===================================================================================================
    # 2. Edge extraction methods

    def method_EDGE(self):
        """
        Extract edge detection features from a given list of images.

        Returns:
        pd.DataFrame: DataFrame containing extracted edge detection features.
        """
        print("============================================")
        print("\033[1mExtracting Edge features\033[0;0m")
        features_list = []

        for img in self.images:
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
