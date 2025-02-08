import os
import sys
import cv2
import numpy as np
import itertools
import pandas as pd
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Classification')))
from Classification import classification_class as classification



class ImagesProcessing:
    def __init__(self, folder_normal, folder_potholes, img_size=(256,256)):
        """
        Initializes the ImagesProcessing class.

        :param folder_normal: Path to the folder containing normal images.
        :type folder_normal: str
        
        :param folder_potholes: Path to the folder containing pothole images.
        :type folder_potholes: str
        
        :param img_size: Tuple representing the size to resize images (width, height).
        :type img_size: tuple, default=(256,256)

        """

        self.img_size = img_size
        self.images_normal = self.load_images_cv2(folder_normal)
        self.images_potholes = self.load_images_cv2(folder_potholes)
        self.labels = [0] * len(self.images_normal) + [1] * len(self.images_potholes)
        self.images = self.images_normal + self.images_potholes



    def load_images_cv2(self, folder_path, gray=True, resize=True):
        """
        Load images from a folder.

        :param folder_path: Path to a folder containing images
        :type folder_path: str

        :param gray: To determine if we load the image in grayscale or in color.
        :type gray: bool, default=True

        :param resize: To determine if we need to resize the images
        :type resize: bool, default=True

        :return images: The list of the images in the folder path
        :rtype: list

        """
        images = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                img_path = os.path.join(folder_path, filename)
                if gray:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    if resize:
                        img = cv2.resize(img, self.img_size)
                    images.append(img)
                        
        return images



    def apply_filter(self, filter_type, kernel_size_gaussian=(5,5), sigma_x=0, d=9, sigma_color=75, sigma_space=75, kernel_size_median=5):
        """
        Apply a specific filter to all images.

        :param filter_type: The specific filter to apply ("gaussian", "bilateral" or "median")
        :rtype filter_type: str

        :param kernel_size_gaussian: Kernel size for the Gaussian filter.
        :type kernel_size_gaussian: tuple, default=(5,5)

        :param sigma_x: Standard deviation in the X direction for the Gaussian filter.
        :type sigma_x: int, default=0

        :param d: Diameter of each pixel neighborhood for bilateral filtering.
        :type d: int, default=9

        :param sigma_color: Filter sigma in the color space for bilateral filtering.
        :type sigma_color: int, default=75

        :param sigma_space: Filter sigma in the coordinate space for bilateral filtering.
        :type sigma_space: int, default=75

        :param kernel_size_median: Kernel size for the Median filter.
        :type kernel_size_median: int, default=5

        """

        filtered_images = []
        for img in self.images:
            if filter_type == 'gaussian':
                filtered_img = cv2.GaussianBlur(img, kernel_size_gaussian, sigma_x)
            elif filter_type == 'median':
                filtered_img = cv2.medianBlur(img, kernel_size_median)
            elif filter_type == 'bilateral':
                filtered_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            else:
                raise ValueError("Invalid filter type. Choose 'gaussian', 'median', or 'bilateral'.")
            filtered_images.append(filtered_img)
        self.images = filtered_images



    def apply_histogram_equalization(self, method='standard', clip_limit=2.0, grid_size=(8,8)):
        """
        Apply Histogram Equalization or Adaptive Histogram Equalization (CLAHE) to enhance contrast.

        :param method: Choose 'standard' for standard histogram equalization or 'clahe' for adaptive histogram equalization.
        :type method: str, default='standard'

        :param clip_limit: Threshold for contrast limiting in CLAHE.
        :type clip_limit: float, default=2.0
        
        :param grid_size: Grid size for CLAHE.
        :type grid_size: tuple, default=(8, 8)
        
        """

        equalized_images = []
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        for img in self.images:
            if len(img.shape) == 2:  # Grayscale image
                if method == 'clahe':
                    equalized_img = clahe.apply(img)
                elif method == 'standard':
                    equalized_img = cv2.equalizeHist(img)
                else:
                    raise ValueError("Invalid method. Choose 'standard' or 'clahe'.")
                equalized_images.append(equalized_img)
            else:
                raise ValueError("Histogram equalization is only applicable to grayscale images.")
        self.images = equalized_images



    def apply_gamma_correction(self, gamma=1.2):
        """
        Apply Gamma Correction to adjust brightness.

        :param gamma: Gamma correction value.
        :type gamma: float, default=1.2

        """
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        self.images = [cv2.LUT(img, table) for img in self.images]
        


    def normalize_image(self):
        """
        Normalize pixel values of all images to the range [0, 1].

        """
        self.images = [img.astype(np.float32) / 255.0 for img in self.images]




    def find_best_preprocessing(self, feature_extraction_method, associated_filter, n_iter=50):
        """
        Finds the best preprocessing configuration for a given feature extraction method and its associated image filter.
        
        :param feature_extraction_method: The function to extract features from images.
        :type feature_extraction_method: function
        
        :param associated_filter: The filter to apply before feature extraction ('gaussian', 'bilateral' or 'median').
        :type associated_filter: str
        
        :param classifier: An instance of the BinaryClassification class.
        :type classifier: BinaryClassification
        
        :return: 
            - best_config (dict): The best preprocessing configuration with its hyperparameters and performance.
            - results (list): A list containing all tested configurations with their corresponding performance.
        :rtype: tuple (dict, list)

        """

        # Define possible hyperparameter values for each filter
        filter_params = {
            "gaussian": {"kernel_size_gaussian": [(3, 3), (5, 5), (7, 7)], "sigma_x": [0, 1, 2]},
            "bilateral": {"d": [5, 9, 12], "sigma_color": [50, 75, 100], "sigma_space": [50, 75, 100]},
            "median": {"kernel_size_median": [3, 5, 7]}
        }
        
        # Add "none" as an option to disable histogram equalization or gamma correction
        histogram_methods = ["none", "standard", "clahe"]
        gamma_values = ["none", 0.8, 1.0, 1.2, 1.5]
        normalization_options = [True, False]  # Whether to apply normalization or not

        # Generate all possible combinations of preprocessing parameters
        all_param_grid = list(itertools.product(
            [tuple(v) for v in zip(*filter_params[associated_filter].values())],  # Regrouper les params du filtre
            histogram_methods,
            gamma_values,
            normalization_options
        ))
        
        # Select a random subset of n_iter configurations
        param_grid = random.sample(all_param_grid, min(n_iter, len(all_param_grid)))

        best_score = -1  # To store the best F1-score obtained
        best_config = None  # To store the best preprocessing configuration
        results = []  # To store all tested configurations

        # Iterate over all preprocessing parameter combinations
        for (param_values), hist_method, gamma, normalize in param_grid:
            # Reset images to their original state
            self.images = self.images_normal + self.images_potholes

            # Apply the selected filter with its corresponding parameters
            if associated_filter == "gaussian":
                self.apply_filter(filter_type="gaussian", kernel_size_gaussian=param_values[0], sigma_x=param_values[1])
            elif associated_filter == "bilateral":
                self.apply_filter(filter_type="bilateral", d=param_values[0], sigma_color=param_values[1], sigma_space=param_values[2])
            elif associated_filter == "median":
                self.apply_filter(filter_type="median", kernel_size_median=param_values[0])

            # Apply histogram equalization only if selected
            if hist_method != "none":
                self.apply_histogram_equalization(method=hist_method)

            # Apply gamma correction only if selected
            if gamma != "none":
                self.apply_gamma_correction(gamma=gamma)

            # Apply normalization only if selected
            if normalize:
                self.normalize_image()

            # Extract features using the given method
            features = feature_extraction_method()

            # Ensure features are formatted as a DataFrame
            if isinstance(features, pd.DataFrame):
                df_features = features
            else:
                df_features = pd.DataFrame(features)

            # Prepare data for classification
            data_processed = classification.DataProcessing(df_features, pd.DataFrame(self.labels), stratified=False)
            env_classifier = classification.BinaryClassification(data_processed, average="macro")

            # Train and evaluate using Logistic Regression
            metrics_results, _, _ = env_classifier.TrainTestLogisticRegression()

            # Retrieve the F1-score on the test set for evaluation
            test_f1_score = metrics_results["f1-score"]["LogReg Test"][0]

            # Store results for analysis
            config_result = {
                "filter": associated_filter,
                "filter_params": param_values,
                "histogram": hist_method,
                "gamma": gamma,
                "normalize": normalize,
                "f1-score": test_f1_score
            }
            results.append(config_result)

            # Update the best configuration if a higher F1-score is found
            if test_f1_score > best_score:
                best_score = test_f1_score
                best_config = config_result

        # Return the best configuration and all results
        return best_config, results  

    def apply_preprocessing(self, config):
        """
        Apply the best preprocessing configuration found.

        :param config: Best configuration containing the filter type, histogram equalization, gamma correction, and normalization.
        :type config: dict
        """
        # Reset images to their original state before applying any preprocessing
        self.images = self.images_normal + self.images_potholes

        # Apply the selected filter if specified in the configuration
        if "filter" in config:
            if config["filter"] == "gaussian":
                # Apply Gaussian blur with the given kernel size and sigma value
                self.apply_filter(filter_type="gaussian", 
                                kernel_size_gaussian=config["filter_params"][0], 
                                sigma_x=config["filter_params"][1])
            elif config["filter"] == "bilateral":
                # Apply Bilateral filter with the specified diameter and sigma values
                self.apply_filter(filter_type="bilateral", 
                                d=config["filter_params"][0], 
                                sigma_color=config["filter_params"][1], 
                                sigma_space=config["filter_params"][2])
            elif config["filter"] == "median":
                # Apply Median blur with the specified kernel size
                self.apply_filter(filter_type="median", kernel_size_median=config["filter_params"][0])

        # Apply histogram equalization if it is not set to "none"
        if "histogram" in config and config["histogram"] != "none":
            # Use either standard histogram equalization or CLAHE (adaptive histogram equalization)
            self.apply_histogram_equalization(method=config["histogram"])

        # Apply gamma correction if a gamma value is provided (not "none")
        if "gamma" in config and config["gamma"] != "none":
            self.apply_gamma_correction(gamma=config["gamma"])

        # Apply normalization if specified in the configuration
        if "normalize" in config and config["normalize"]:
            self.normalize_image()
            

        self.images = [
            (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img  # Convert to uint8
            for img in self.images
        ]

        self.images = [
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img  # Convert to grayscale if needed
            for img in self.images
        ]