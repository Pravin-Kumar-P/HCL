import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load & Preprocess
# -----------------------------
img = cv2.imread(r"C:\Users\OMEN\Downloads\images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smooth but keep edges (mimics dye-penetrant developer)
smooth = cv2.bilateralFilter(gray, 7, 50, 50)

# Black-hat -> highlights cracks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)

# Contrast enhancement
blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

# -----------------------------
# Adaptive Threshold (more NDT-like)
# -----------------------------
th = cv2.adaptiveThreshold(
    blackhat,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,  # window size
    -10  # bias
)

# Remove noise
clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# -----------------------------
# Crack Skeleton (1-pixel lines)
# -----------------------------
skeleton = cv2.ximgproc.thinning(clean)

# -----------------------------
# Show NDT view
# -----------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original"); plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(blackhat, cmap='gray')
plt.title("BlackHat Contrast"); plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(clean, cmap='gray')
plt.title("Crack Segmentation"); plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(skeleton, cmap='gray')
plt.title("Crack Skeleton (NDT)"); plt.axis("off")

plt.tight_layout()
plt.show()
