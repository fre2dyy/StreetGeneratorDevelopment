import numpy as np
import cv2

def deginrad(degree):
    import numpy as np

    radiant = 2*np.pi/360 * degree
    return radiant

# #################################################################################
# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, phi, ktype)
# ksize - size of gabor filter (n, n) --> line width of road markings in pixel
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# phi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold
# #################################################################################
#
# all values except theta (orientation of the street) can remain unchanged
#
# #################################################################################


theta = deginrad(-67)   # unit circle: right: 90 deg, left: -90 deg, straight: 0 deg
g_kernel = cv2.getGaborKernel((6, 6), 10, theta, 5, 0, 0, ktype=cv2.CV_32F)

# img = cv2.imread('files/motorway/colour/motorway_clipped_S.tiff')
# img = cv2.imread('files/motorway/colour/motorway_dop.tif.')
img = cv2.imread('files/motorway/gamma/streets_gamma0.45.tiff')
# img = cv2.imread('files/motorway/gamma/streets_gamma0.35.tiff')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

# cv2.imshow('image', img)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 800,800)

# cv2.imshow('filtered image', filtered_img)
# cv2.resizeWindow('filtered image', 800,800)


h, w = g_kernel.shape[:2]
g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
cv2.imshow('gabor kernel (resized)', g_kernel)

# cv2.imwrite("files/motorway/gabor/motorway_gabor.tiff", filtered_img)
cv2.imwrite("files/motorway/gabor/streets_gabor.tiff", filtered_img)


cv2.waitKey(0)
cv2.destroyAllWindows()