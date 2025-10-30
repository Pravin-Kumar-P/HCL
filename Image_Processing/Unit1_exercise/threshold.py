import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("Sc_120.png", 0)
def manual_otsu_threshold(image):
    # Compute histogram 
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size

    # Normalize histogram (probability of each intensity)
    prob = hist / total_pixels

    current_max, threshold = 0, 0
    sum_total, sum_background, weight_background = 0, 0, 0

    # Sum of all intensity * probability
    for i in range(256):
        sum_total += i * prob[i]

    # Try every possible threshold value
    for t in range(256):
        weight_background += prob[t]                # weight of background
        if weight_background == 0:
            continue

        weight_foreground = 1 - weight_background   # weight of foreground
        if weight_foreground == 0:
            break

        sum_background += t * prob[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Between-class variance formula
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # Keep track of the maximum variance
        if var_between > current_max:
            current_max = var_between
            threshold = t

    print(f"✅ Otsu’s optimal threshold = {threshold}")

    # Apply threshold
    otsu_result = (image >= threshold).astype(np.uint8) * 255
    return otsu_result


# -----------------------------
# 2️⃣ ADAPTIVE THRESHOLDING (Manual)
# -----------------------------
def manual_adaptive_threshold(image, block_size=11, C=2):
    """Apply adaptive thresholding using local mean within a window."""
    padded = cv2.copyMakeBorder(image, block_size//2, block_size//2,
                                block_size//2, block_size//2, cv2.BORDER_REFLECT)

    adaptive_result = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local region
            local_region = padded[i:i+block_size, j:j+block_size]
            local_mean = np.mean(local_region)

            # Threshold rule: pixel >= mean - C → white
            if image[i, j] > (local_mean - C):
                adaptive_result[i, j] = 255
            else:
                adaptive_result[i, j] = 0

    return adaptive_result


otsu_thresh = manual_otsu_threshold(img)
adaptive_thresh = manual_adaptive_threshold(img, block_size=11, C=2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title(" Otsu")

plt.subplot(1, 3, 3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title(" Adaptive")

plt.show()
