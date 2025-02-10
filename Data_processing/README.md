Multiple processing methods can and are performed on images before applying Feature Extraction techniques.


# I. Filters

## 1. Gaussian Filter

The **Gaussian filter** is a linear filter used to smooth an image and reduce noise, while preserving the edges to some extent. It convolves the image using a Gaussian kernel, which allows to give more weights to pixels closer to the center because of the Gaussian distribution. It can be seen therefore as an improvement of the mean filter.


## 2. Median Filter

The **Median filter**, a non-linear filter, preserves the edges better than the Gaussian filter. The value of a pixel is replaced by the median value of its $N \times N$ neighborhood. It is the best choice to **remove impulse noise** (salt-and-pepper noise).


## 3. Bilateral Filter

The **Bilateral filter** is a non linear filter that smooths an image while **preserving edges** better than the two previous filters. Indeed, while it also uses the Gaussian kernel, it uses an extra component, a weighting function based on intensity similarity, to ensure that edges are not blurred.


# II. Histogram Equalization

Histogram Equalization **enhances contrast** by redistributing pixel intensity values to use the full range of intensities. However, it should be used carefully, as in some cases, it might degrade the image.
