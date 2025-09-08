import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("E:/HCL/archive/wood/test/hole/001.png")
#Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Denoising using Gausianblur
blur = cv2.GaussianBlur(gray, (5,5), 0)
#Edge detection using Cannyedge
edges = cv2.Canny(blur, threshold1=50, threshold2=150)
#Morphological operations to close small gaps
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(edges, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# binary ground truth mask (0 = background, 255 = crack)
ground_truth = np.zeros_like(gray)
ground_truth[mask > 0] = 255

plt.figure(figsize=(12,5))
#to display
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Detected Edges (Crack)")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Ground Truth Mask (Binary)")
plt.imshow(ground_truth, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
# to save
cv2.imwrite("ground_truth_mask.png", ground_truth)
