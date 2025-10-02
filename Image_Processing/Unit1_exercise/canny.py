import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_blur(image, size=5, sigma=1.4):
    k = size // 2
    kernel = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)   # normalize

    # Convolution
    h, w = image.shape
    pad = size // 2
    padded = np.pad(image, pad, mode="edge")
    result = np.zeros_like(image, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            result[i, j] = np.sum(region * kernel)
    return result

def sobel_gradients(image):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    h, w = image.shape
    padded = np.pad(image, 1, mode="edge")
    Gx = np.zeros_like(image, dtype=float)
    Gy = np.zeros_like(image, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            Gx[i, j] = np.sum(region * Kx)
            Gy[i, j] = np.sum(region * Ky)
    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    h, w = magnitude.shape
    result = np.zeros((h, w), dtype=float)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180  # convert to [0, 180)

    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            # Angle 0°
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45°
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90°
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135°
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                result[i, j] = magnitude[i, j]
            else:
                result[i, j] = 0
    return result

def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    high_threshold = img.max() * high_ratio
    low_threshold = img.max() * low_ratio
    result = np.zeros_like(img, dtype=np.uint8)

    strong = 255
    weak = 75

    strong_edges = (img >= high_threshold)
    weak_edges = ((img >= low_threshold) & (img < high_threshold))

    result[strong_edges] = strong
    result[weak_edges] = weak

    return result

def hysteresis(img):
    h, w = img.shape
    strong = 255
    weak = 75

    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i, j] == weak:
                # If any neighbor is strong, promote to strong
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img
# Load image (grayscale)
img = cv2.imread("butterfly.jpg", cv2.IMREAD_GRAYSCALE)
# Step 1: Gaussian smoothing
smoothed = gaussian_blur(img, size=5, sigma=1.4)
# Step 2: Gradient magnitude + direction
magnitude, direction = sobel_gradients(smoothed)
# Step 3: Non-maximum suppression
nms = non_max_suppression(magnitude, direction)
# Step 4: Double threshold 
thresholded = double_threshold(nms)
# Step 5: Edge tracking by hysteresis
final_edges = hysteresis(thresholded)


plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2, 3, 2), plt.imshow(smoothed, cmap='gray'), plt.title("1. Gaussian Blur")
plt.subplot(2, 3, 3), plt.imshow(magnitude, cmap='gray'), plt.title("2. Gradient Magnitude")
plt.subplot(2, 3, 4), plt.imshow(nms, cmap='gray'), plt.title("3. Non-Max Suppression")
plt.subplot(2, 3, 5), plt.imshow(thresholded, cmap='gray'), plt.title("4. Double Threshold")
plt.subplot(2, 3, 6), plt.imshow(final_edges, cmap='gray'), plt.title("5. Hysteresis (Final Edges)")
plt.tight_layout()
plt.show()