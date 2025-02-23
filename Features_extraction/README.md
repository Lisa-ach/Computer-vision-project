Multiple feature extraction algorithms are used. Their implementation can be found in the class FeatureExtraction of the file *feature_extraction_class.py*. The file *feature_extraction.ipynb* allows to find the best hyperparameters for each method.

The methods used can be listed as follows.

# I. Low-Level Features

Low-Level Feature Extraction techniques focus on basic image characteristics like **edges, corners, or points of interest.**

## 1. Interest point detection algorithms

Interest point detection algorithms focus on finding distinctive points that are easy to match across images.

- **SIFT (Scale-Invariant Feature Transform), 1999**

SIFT first applies a convolution to all the images, with different Gaussian kernels, using different scales $\sigma_n = k^n \sigma_0$ (usually $k=\sqrt{2}$). If $I(x,y)$ is the original image and $G(x,y,\sigma)$ is the Gaussian function, then the blurred version of the image can be thus defined as:

$$L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$$

The **Difference of Gaussians (DoG)** is then computed by subtracting images from two consecutive scales:

$$D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$$

Keypoints are defined as local extrema in the DoG images. To find them, each pixel in a DoG image is compared to its 26 neighbors: 8 pixels from the same image, 9 pixels in the previous scale and 9 pixels in the next scale. It is considered a keypoint if this point is an extremum in this neighborhood. Some techniques are also used afterwards to remove some unstable or weak keypoints such as the one along edges.

Once the keypoints are detected, **descriptors** are created to describe them and so that they can be matched across different images. To define them, for each pixel belonging to a $16 \times 16$ grid around a keypoint, are first computed the gradient magnitude $M$ and orientation $\theta$ using finite differences. If we define $G_x = L(x+1,y) - L(x-1,y)$ and $G_y = L(x,y+1) - L(x,y-1)$ the gradients respectively in X and Y direction, then magnitude and orientation can defined as:

$$M(x,y) = \sqrt{G_x^2 + G_y^2} ; \theta(x,y) = tan^{-1} (\frac{G_y}{G_x})$$

The $16 \times 16$ grid is divided into 16 smaller $4 \times 4$ cells so that each $4 \times 4$ region has its own orientation histogram, each containing 8 orientation bins. Therefore, each cell gives 8 orientation bins, and consequently the keypoint descriptor contains $8 \times 16 = 128$ values. 


- **ORB (Oriented FAST and Rotated BRIEF), 2011**

ORB is a fast and free **alternative to SIFT and SURF**. It uses the **FAST algorithm** to detect keypoints. FAST identifies corners by analyzing around each pixel a small circular neighborhood of 16 pixels. If $N$ ($N = 9$ typically) contiguous pixels in the circle are all darker or brighter by a threshold compared to a pixel, then the pixel is considered a corner. FAST is efficient as instead of checking all pixels in the circle, it first checks if 4 of these pixels meet the condition, and if it is the case, the entire circle is checked.

- **Harris corner detection**

Harris corner detection is an algorithm that allows to identify corners in an image. Corners are points where there are significant changes in intensity in **multiple directions**, making them useful for feature detection and tracking.

First, it computes the matrix M, defined as:

$$
M =
\sum\limits_{x,y} 
w(x,y)
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
$$

where $I_x$ and $I_y$ are derivatives  of the image in the X and Y direction:
$$
I_x = \frac{\partial I}{\partial x}, \quad I_y = \frac{\partial I}{\partial y};
$$

$
w(x,y)$ is a window function (Gaussian for instance) that gives more weight to central pixels.


It then computes Harris corner score:

$$R = det(M) - k trace(M)^2$$

Typically k is equal to 0.04 - 0.06. The values of R are positive near a corner, negative near a contour, and low in a region of constant intensity.

## 2. Edge detection algorithms

Edge detection algorithms focus on identifying object boundaries where there are significant changes in intensity. These methods help extract key features from images. Edge detection can be divided into **gradient-based methods** and **second derivative methods**.

### a) Gradient-Based Edge Detectors

Gradient-based methods compute the **first derivative** of the image to detect areas of high intensity change. They are **effective at finding edges** but tend to be **sensitivity to noise**.

- **Canny Edge Detector (1986)**

The **Canny Edge Detector** is a popular method for detecting edges. It first applies **Gaussian smoothing** to reduce noise. Then, it computes the **image gradient** to find intensity changes. After that, **non-maximum suppression** is used to thin the edges. Finally, a **double thresholding** step classifies edges as strong or weak, and weak edges connected to strong ones are preserved.

- **Sobel Operator (1968)**

The **Sobel operator** highlights edges by applying horizontal and vertical filters to detect changes in intensity. 

The gradient magnitude is computed as $$G=\sqrt{G_x^2+G_y^2}$$ where $G_x$ and $G_y$ are gradients computed using Sobel kernels:

$$G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
, \quad
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}$$

The Sobel operator is computationally **efficient and simple** but **sensitive to noise**, making it less reliable in high-noise environments.

- **Prewitt Operator (1970)**

The **Prewitt operator** is similar to Sobel but uses equal weights in the convolution kernels: 

$$G_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
, \quad
G_y =
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}$$

It detects edges effectively but is slightly less accurate in identifying sharp intensity changes.

- **Scharr Operator (2000)**

The **Scharr operator** is an optimized version of Sobel that **reduces noise sensitivity** while maintaining sharp edge detection by improving **gradient estimation accuracy**. The convolution kernels are: 

$$G_x =
\begin{bmatrix}
-3 & 0 & 3 \\
-10 & 0 & 10 \\
-3 & 0 & 3
\end{bmatrix}
, \quad
G_y =
\begin{bmatrix}
3 & 10 & 3 \\
0 & 0 & 0 \\
-3 & -10 & -3
\end{bmatrix}$$


- **Roberts Cross Operator (1963)**

The **Roberts operator** detects edges by calculating the difference between diagonally adjacent pixels. 

$$G_x =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
, \quad
G_y =
\begin{bmatrix}
0 & 1  \\
-1 & 0 
\end{bmatrix}$$

It is **simple and fast** but not as precise as more advanced techniques due to its **high sensitive to noise**.


### b) Second-Derivative Edge Detectors

Second-derivative methods compute the **Laplacian**, which highlights regions of **rapid intensity change.**

- **Laplacian Operator**

The **Laplacian operator** detects edges by computing the **second derivative** of the image, which enhances areas of rapid intensity change. The Laplacian Kernel is:

$$L =
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0 
\end{bmatrix}$$


### c) Adaptive Edge Detection Techniques

These methods adapt thresholding techniques to improve edge detection in complex scenarios.

- **Adaptive Thresholding**

Adaptive thresholding dynamically adjusts threshold values based on **local intensity variations**, improving edge segmentation, especially in images with **non-uniform lighting conditions**.

- **Local Binary Patterns (LBP)**

LBP is used for texture-based edge analysis. It encodes pixel intensity variations in a small neighborhood, making it useful for capturing edge and texture information simultaneously.


| Algorithm          | Type                | Strengths                                      | Weaknesses                                |
|--------------------|---------------------|-----------------------------------------------|-------------------------------------------|
| **Canny**         | Gradient-Based      | Accurate edges, noise reduction               | Computationally expensive                 |
| **Sobel**         | Gradient-Based      | Simple and efficient                          | Sensitive to noise                        |
| **Prewitt**       | Gradient-Based      | Less computation than Sobel                   | Less precise                              |
| **Scharr**        | Gradient-Based      | Improved precision over Sobel                 | Computationally expensive                 |
| **Laplacian**     | Second-Derivative   | Detects fine details                          | High noise sensitivity                    |
| **Roberts**       | Gradient-Based      | Fast, simple                                 | Poor accuracy, very sensitive to noise   |
| **Adaptive Thresholding** | Adaptive   | Works on uneven lighting                     | Hard to tune parameters                   |
| **Local Binary Patterns (LBP)** | Texture-Based | Effective for texture analysis, robust to illumination changes | Not ideal for pure edge detection |

# II. Structural Features

Structural Feature Extraction techniques focus on capturing the **shape or geometry of objects in an image.**

The **Histogram of Oriented Gradients (HOG)** is one of these techniques. It divides the image in cells, and for each pixel of this cell it computes the gradient magnitude and the orientation. It then computes for each cell a histogram of the oriented gradients. These histograms represent the distribution of edge directions in each cell, forming a set of local feature descriptors. 

HoG is typically used to  describe  larger  image  regions,  whereas  SIFT,  that  also uses orientations, focuses on keypoints.



# III. Texture Features

Texture-based Feature Extraction methods focus on analyzing patterns, textures and fine surface details. 

**Gabor filters** are among these methods. Gabor filters apply a convolution kernel to the image. Different Gabor filters are used to extract the useful features from an image, each one focusing on a special frequency and orientation. The kernel can be defined as:

$$
G(x,y) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi f x' + \phi\right)
$$

$$
\text{with} \: x' = x \cos\theta + y \sin\theta, \quad y' = -x \sin\theta + y \cos\theta
$$



# References

David G. Lowe. (2004) *Distinctive Image Features from Scale-Invariant Keypoints.* International Journal of Computer Vision 

Edward Rosten and Tom Drummond. (2006) *Machine Learning for High-Speed Corner Detection.* European Conference on Computer Vision

Chris Harris and Mike Stephens. (1988) *A Combined Corner and Edge Detector.*

John Canny (1986) *A Computational Approach to Edge Detection.*

I. Sobel and G. Feldman (1968) *An Isotropic 3x3 Image Gradient Operator.*

Judith M. S. Prewitt (1970) *Object Enhancement and Extraction*.

D. Marr and E.Hildreth (1980) *Theory of Edge Detection*.

R. C. Gonzalez and R.E. Woods (2002) *Digital Image Processing*.

T. Ojala, M. Pietikäinen and T. Mäenpää (1996) *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*.

H. Scharr (2000) *Optimal Operators in Digital Image Processing*





