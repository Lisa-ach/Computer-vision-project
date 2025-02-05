import cv2
import numpy as np

def gaussian_filter(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Filtering to reduce noise while preserving important features.
    
    :param image: Input image (grayscale or BGR).
    :param kernel_size: Kernel size for Gaussian blur.
    :return: Denoised image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def histogram_equalization(image):
    """
    Apply Histogram Equalization to adjust image intensity distribution for better contrast.
    
    :param image: Input grayscale image.
    :return: Image with enhanced contrast.
    """
    return cv2.equalizeHist(image)

def preprocess_image(image):
    """
    Perform data preprocessing on the input image.
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian Filtering (Noise Reduction & Smoothing)
    3. Apply Histogram Equalization (Illumination Adjustment & Enhancement)
    
    :param image: Input BGR image.
    :return: Preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    smoothed = gaussian_filter(gray)  # Noise Reduction & Smoothing
    enhanced = histogram_equalization(smoothed)  # Illumination Adjustment
    return enhanced
