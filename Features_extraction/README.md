Multiple feature extraction algorithms are used.

# I. Interest point detection algorithms

## 1. Grayscale

- **SIFT (Scale-Invariant Feature Transform), 1999**

SIFT first applies a convolution to all the images, with different Gaussian kernels, using different scales $\sigma_n = k^n \sigma_0$ (usuallly $k=\sqrt{2}$). If $I(x,y)$ is the original image and $G(x,y,\sigma)$ is the Gaussian function, then the blurred version of the image can be thus defined as:

$$L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$$

The **Difference of Gaussians (DoG)** is then computed by subtracting images from two consecutive scales:

$$D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$$

Key points are defined as local extrema in the DoG images. To find them, each pixel in a DoG image is compared to its 26 neighbors: 8 pixels from the same image, 9 pixels in the previous scale and 9 pixels in the next scale. It is considered a key point if this point is an extremum in this neighborhood. Some techniques are also used afterwards to remove some unstable or weak keypoints such as the one along edges.

Once the key points are detected, **descriptors** are created to describe them and so that they can be matched across different images. To define them, for each pixel belonging to a $16 \times 16$ grid around a key point, are first computed the gradient magnitude $M$ and orientation $\theta$ using finite differences. If we define $G_x = L(x+1,y) - L(x-1,y)$ the gradient in X direction, and $G_y = L(x,y+1) - L(x,y-1)$, then they can defined as:

$$M(x,y) = \sqrt{G_x^2 + G_y^2} ; \theta(x,y) = tan^{-1} (\frac{G_y}{G_x})$$

The $16 \times 16$ grid is divided into 16 smaller $ 4 \times 4$ cells so that each $4 \times 4$ region has its own orientation histogram, each containing 8 orientation bins. Therefore, each cell gives 8 orientation bins, and consequently the key point descriptor contains $8 \times 16 = 128$ values. 


- **ORB (Oriented FAST and Rotated BRIEF), 2011**

## 2. Color



# References

David G. Lowe. (2004) Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 



