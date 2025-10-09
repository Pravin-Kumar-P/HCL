import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('peppers.png')  
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to HSV and LAB color spaces
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# Red color segmentation in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Apply mask on original RGB image
seg_hsv = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# Plot all results
plt.figure(figsize=(15, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original RGB')
plt.axis('off')

# HSV image
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV Image')
plt.axis('off')

# LAB image
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB))
plt.title('LAB Image')
plt.axis('off')

# HSV Channels
h, s, v = cv2.split(hsv)
plt.subplot(2, 3, 4)
plt.imshow(h, cmap='hsv')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

# Segmented result
plt.subplot(2, 3, 6)
plt.imshow(seg_hsv)
plt.title('Segmented (Red Color)')
plt.axis('off')

plt.tight_layout()
plt.show()
