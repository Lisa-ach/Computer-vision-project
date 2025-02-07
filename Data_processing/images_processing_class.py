import os
import cv2
import numpy as np



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


