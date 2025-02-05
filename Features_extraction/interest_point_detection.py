import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_processing')))
# import import_images as i

def method_SIFT (images):
    # 1. Extract SIFT features from all images
    print("=========1. Extract SIFT features from all images=========")
    sift = cv2.SIFT_create()
    descriptors_list = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
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
    print("=========2. Stack all descriptors for clustering (BoVW) =========")
    all_descriptors = np.vstack(descriptors_list)

    # 3. Cluster descriptors using KMeans (BoVW approach)
    print("=========3. Cluster descriptors using KMeans (BoVW approach) =========")
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # 4. Create feature histograms for each image
    print("========= 4. Create feature histograms for each image =========")
    def extract_features(image_descriptors, kmeans_model, num_clusters):
        feature_histogram = np.zeros(num_clusters)
        if image_descriptors is not None:
            labels = kmeans_model.predict(image_descriptors)
            for label in labels:
                feature_histogram[label] += 1
        return feature_histogram

    feature_vectors = [extract_features(desc, kmeans, num_clusters) for desc in descriptors_list]

    return feature_vectors

def method_ORB(images):
    # 1. Extract ORB features from all images
    print("=========1. Extract ORB features from all images=========")
    orb = cv2.ORB_create()
    descriptors_list = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the image
            # plt.figure(figsize=(8, 6))
            # plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
            # plt.axis('off')
            # plt.title(f"ORB Features ({len(keypoints)} Keypoints)")
            # plt.show()

    # 2. Stack all descriptors for clustering (BoVW)
    print("=========2. Stack all descriptors for clustering (BoVW) =========")
    all_descriptors = np.vstack(descriptors_list)

    # 3. Cluster descriptors using KMeans (BoVW approach)
    print("=========3. Cluster descriptors using KMeans (BoVW approach) =========")
    num_clusters = 5  # You can tune this parameter
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # 4. Create feature histograms for each image
    print("========= 4. Create feature histograms for each image =========")
    def extract_features(image_descriptors, kmeans_model, num_clusters):
        feature_histogram = np.zeros(num_clusters)
        if image_descriptors is not None:
            labels = kmeans_model.predict(image_descriptors)
            for label in labels:
                feature_histogram[label] += 1
        return feature_histogram

    feature_vectors = [extract_features(desc, kmeans, num_clusters) for desc in descriptors_list]
    return feature_vectors