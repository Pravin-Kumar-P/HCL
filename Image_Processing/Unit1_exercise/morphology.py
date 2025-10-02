import cv2
import numpy as np
import matplotlib.pyplot as plt
mask = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Mask")
plt.imshow(mask, cmap='gray')
plt.axis("off")
plt.subplot(1, 3, 2)
plt.title("After Opening")
plt.imshow(opened, cmap='gray')
plt.axis("off")
plt.subplot(1, 3, 3)
plt.title("After Closing")
plt.imshow(closed, cmap='gray')
plt.axis("off")

plt.show()
