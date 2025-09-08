import cv2
import numpy as np
#to read image in grayscale 
image = cv2.imread('profile.jpg', cv2.IMREAD_GRAYSCALE)
kernel_size = 25
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
#low-pass filter
low_pass_filtered_image = cv2.filter2D(image, -1, kernel)
cv2.imshow('Original Image', image)
cv2.imshow('Low pass filtered image', low_pass_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
