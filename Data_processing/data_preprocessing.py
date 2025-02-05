import cv2
import numpy as np

def gaussian_filter(image, kernel_size=(5, 5)):
    """Apply Gaussian Filtering to reduce noise while preserving important features."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply Bilateral Filtering to reduce noise while preserving edges."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def median_filter(image, kernel_size=5):
    """Apply Median Filtering to reduce salt-and-pepper noise while preserving edges."""
    return cv2.medianBlur(image, kernel_size)

def histogram_equalization(image):
    """Apply Histogram Equalization to adjust image intensity distribution for better contrast."""
    return cv2.equalizeHist(image)

def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def gamma_correction(image, gamma=1.2):
    """Apply Gamma Correction to adjust brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def normalize_image(image):
    """Normalize pixel values to range [0, 1]."""
    return image.astype(np.float32) / 255.0

def preprocess_image(image, method="gaussian", equalization="hist", normalize=True, gamma=None):
    """
    Perform data preprocessing on the input image.

    Steps:
    1. Convert to grayscale
    2. Apply Noise Reduction (Gaussian, Bilateral, or Median Filtering)
    3. Apply Contrast Enhancement (Histogram Equalization or CLAHE)
    4. Apply Gamma Correction (optional)
    5. Normalize (optional)
    
    :param image: Input BGR image.
    :param method: Noise reduction method ("gaussian", "bilateral", "median").
    :param equalization: Contrast enhancement ("hist" for equalization, "clahe" for CLAHE).
    :param normalize: Whether to normalize pixel values to [0,1].
    :param gamma: Gamma correction factor (if None, not applied).
    :return: Preprocessed image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Noise Reduction
    if method == "bilateral":
        smoothed = bilateral_filter(gray)
    elif method == "median":
        smoothed = median_filter(gray)
    else:
        smoothed = gaussian_filter(gray)

    # Apply Contrast Enhancement
    if equalization == "clahe":
        enhanced = adaptive_histogram_equalization(smoothed)
    else:
        enhanced = histogram_equalization(smoothed)

    # Apply Gamma Correction (if specified)
    if gamma is not None:
        enhanced = gamma_correction(enhanced, gamma)

    # Normalize if required
    if normalize:
        enhanced = normalize_image(enhanced)

    return enhanced
