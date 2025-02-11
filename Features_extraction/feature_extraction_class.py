import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

from skimage.filters import prewitt, roberts, laplace, threshold_otsu, scharr
from skimage.feature import local_binary_pattern

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Classification')))
from Classification import classification_class as classification


class FeatureExtraction:

    def __init__(self, imagesClass, metric="accuracy", average="binary"):
        """Initialization of the class

        :param imagesClass: object of class ImagesProcessing
        :type imagesClass: ImagesProcessing
        
        :param metric: metric to use in the class
        :type metric: str, default="accuracy"

        :param average: parameter average for f1-score, precision and recall
        :type metric str, default="binary"
        """
        self.imagesClass = imagesClass
        self.images = imagesClass.images
        self.method_functions = {
            'SIFT': self.method_SIFT,
            'ORB':  self.method_ORB,
            'Harris' : self.method_Harris,
            'EDGE': self.method_EDGE,
            'Otsu': self.method_otsu,
            'Adaptive': self.method_adaptive,
            'Gabor': self.method_adaptive,
            'LBP': self.method_LBP,
        }
        self.methods = {'SIFT', 'ORB', 'Harris', 'EDGE', 'Otsu', 'Adaptive', 'Gabor', 'LBP'}
        # self.metric = metric # to be used eventually for choosing hyperparameters
        # self.average = average
        
    # ===================================================================================================
    # 1. Interest points detection methods
        
    def method_SIFT (self, num_clusters=5, nfeatures=500, nOctaveLayers=3, sigma=1.6):
        print("============================================")
        print("\033[1mExtracting SIFT Features\033[0;0m")
        # 1. Extract SIFT features from all images
        sift = cv2.SIFT_create(nfeatures=nfeatures,
                               nOctaveLayers=nOctaveLayers,
                               sigma=sigma)
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

    def method_ORB(self, num_clusters=5, nfeatures=500, scaleFactor=1.2):
        print("============================================")
        print("\033[1mExtracting ORB Features\033[0;0m")
        # 1. Extract ORB features from all images
        orb = cv2.ORB_create(nfeatures=nfeatures,
                             scaleFactor=scaleFactor)
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
    
    def method_Harris(self, blockSize=2, ksize=3, k=0.04, threshold=0.01):
        print("============================================")
        print("\033[1mExtracting Harris Corner Features\033[0;0m")
        features_list = []
        
        for img in self.images:
            # Check if the image is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # Convert image to float32 as required by cv2.cornerHarris
            gray = np.float32(gray)
            # Apply Harris Corner Detection
            dst = cv2.cornerHarris(gray, blockSize, ksize, k)
            # Dilate to mark the corners
            dst = cv2.dilate(dst, None)
            # Threshold to detect strong corners
            corner_mask = dst > threshold * dst.max()
            
            # Compute features:
            # - count of corners
            # - mean and standard deviation of corner responses
            # - mean and standard deviation of the corner positions (x and y)
            count = np.sum(corner_mask)
            if count > 0:
                indices = np.argwhere(corner_mask)
                responses = dst[corner_mask]
                mean_response = np.mean(responses)
                std_response = np.std(responses)
                mean_x = np.mean(indices[:, 1])  # x coordinate (column index)
                std_x = np.std(indices[:, 1])
                mean_y = np.mean(indices[:, 0])  # y coordinate (row index)
                std_y = np.std(indices[:, 0])
            else:
                mean_response = 0
                std_response = 0
                mean_x = 0
                std_x = 0
                mean_y = 0
                std_y = 0
            
            features_list.append([count, mean_response, std_response, mean_x, std_x, mean_y, std_y])
        
        return features_list

    # ===================================================================================================
    # 2. Edge extraction methods

    def method_EDGE(self, canny_threshold1=100, canny_threshold2=200,
                    sobel_ksize=3, laplacian_ksize=3):
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
            # otsu_thresh = threshold_otsu(gray)
            # Canny
            # edges_canny = cv2.Canny(gray, otsu_thresh * 0.5, otsu_thresh * 1.5)
            edges_canny = cv2.Canny(gray, canny_threshold1, canny_threshold2)
            
            # Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            
            prewitt_edges = prewitt(gray)
            laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=laplacian_ksize)
            # laplacian_edges = laplace(gray)
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
    

    # ===================================================================================================

    def method_otsu(self):

        print("============================================")
        print("\033[1mData Segmentation using Otsu's Thresholding\033[0;0m")
        features_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
        
            # Apply Otsu's Thresholding
            _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
            # Extract histogram features from binary image
            hist = cv2.calcHist([binary_image], [0], None, [256], [0, 256])
            # Normalize the histogram
            hist = hist.flatten() / np.sum(hist)  
            features_list.append(hist)


        return features_list
    
    def method_adaptive(self):

        print("============================================")
        print("\033[1mData Segmentation using Adaptive's Thresholding\033[0;0m")
        features_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
        
            # Adaptive Thresholding (Gaussian)
            adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
            
            hist_adaptive = cv2.calcHist([adaptive_binary], [0], None, [256], [0, 256]).flatten()
            hist_adaptive = hist_adaptive / np.sum(hist_adaptive)

            features_list.append(hist_adaptive)

        return features_list
    
    
    def method_Gabor(self, ksize=7, sigma=4.0, lambd=10.0, gamma=0.5):

        print("============================================")
        print("\033[1mExtracting Surface Textures Features using Gabor filters\033[0;0m")
        features_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
        
            # Define Gabor filter orientations
            orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            features = []
            for theta in orientations:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)  # Apply Gabor filter
                
                # Extract statistical features
                mean_val = np.mean(filtered_img)
                var_val = np.var(filtered_img)
                energy = np.sum(filtered_img**2)

                features.extend([mean_val, var_val, energy])

            features_list.append(np.array(features))

        return features_list

    def method_LBP(self,radius=3, num_points=24):

        print("============================================")
        print("\033[1mExtracting Spatial Texture Features using LBP\033[0;0m")
        features_list = []

        for img in self.images:
            # Check if it is not already in grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            else:
                gray = img
        
            # Compute LBP
            lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
            
            # Compute histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
            
            # Normalize histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Avoid division by zero

            features_list.append(hist)

        return features_list

    def optimal_hyperparameters(self, methods=('SIFT', 'ORB', "Harris", 'EDGE')):

        hyperparameters = {
            'SIFT': {
                'num_clusters':      [5, 10],
                'nfeatures':         [300, 500],
                'nOctaveLayers':     [3, 4],
                # 'contrastThreshold': [0.04, 0.06],
                # 'edgeThreshold':     [10, 20],
                'sigma':             [1.2, 1.6]
            },
            'ORB': {
                'num_clusters':   [5, 10],
                'nfeatures':      [300, 500],
                'scaleFactor':    [1.2, 1.5],
                # 'nlevels':        [8, 12],
                # 'edgeThreshold':  [15, 31],
                # 'fastThreshold':  [10, 20]
            },
            'Harris':{
                'blockSize':[2,3],
                'ksize':[3,4],
                'k':[0.04, 0.05],
                'threshold':[0.01, 0.02]
            },
            'EDGE': {
                'canny_threshold1': [50, 100],
                'canny_threshold2': [150, 200],
                'sobel_ksize':      [3, 5],
                'laplacian_ksize':  [3, 5]
            }
        }

        best_configs = {m: [] for m in methods}
        
        for method in methods:

            best_score=0
            
            func = self.method_functions[method]

            param_grid = hyperparameters[method]
            for params in ParameterGrid(param_grid):
            
                df_features =  pd.DataFrame(func(**params))

                data_processed = classification.DataProcessing(df_features, pd.DataFrame(self.imagesClass.labels), stratified=False)
                env_classifier = classification.BinaryClassification(data_processed, average="macro")

                # Train and evaluate using Logistic Regression
                metrics_results, _, _ = env_classifier.TrainTestLogisticRegression()

                test_f1_score = metrics_results["f1-score"]["LogReg Test"][0]

                # Compare
                if test_f1_score > best_score:
                    best_score = test_f1_score
                    best_config = params


            best_configs[method] = best_config
                
        return best_configs