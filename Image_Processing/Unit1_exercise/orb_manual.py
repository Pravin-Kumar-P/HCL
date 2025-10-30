import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def detect_keypoints(image, threshold=30):
    h, w = image.shape
    keypoints = []
    offsets = np.array([
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ])

    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = int(image[y, x])
            circle = np.array([int(image[y + dy, x + dx]) for dx, dy in offsets])
            brighter = np.sum(circle > center + threshold)
            darker = np.sum(circle < center - threshold)
            if brighter >= 12 or darker >= 12:
                keypoints.append((x, y))
    return keypoints

def compute_orientation(image, keypoints, patch_size=31):
    orientations = []
    half = patch_size // 2
    for (x, y) in keypoints:
        if x < half or y < half or x >= image.shape[1] - half or y >= image.shape[0] - half:
            orientations.append(0)
            continue

        patch = image[y - half:y + half + 1, x - half:x + half + 1]
        total_pixel_sum = np.sum(patch)
        if total_pixel_sum == 0:
            orientations.append(0)
            continue

        # Corrected moment-based orientation
        m10 = np.sum(np.arange(-half, half + 1) * np.sum(patch, axis=0))
        m01 = np.sum(np.arange(-half, half + 1)[:, np.newaxis] * np.sum(patch, axis=1)[:, np.newaxis])
        angle = math.atan2(m01, m10)
        orientations.append(angle)
    return orientations
def generate_brief_descriptors(image, keypoints, orientations, patch_size=31, n_bits=256):
    np.random.seed(42)
    half = patch_size // 2
    pattern = np.random.randint(-half, half, (n_bits, 4))
    descriptors = []
    h, w = image.shape

    for (x, y), theta in zip(keypoints, orientations):
        if x < half or y < half or x >= w - half or y >= h - half:
            descriptors.append(np.zeros(n_bits, dtype=np.uint8))
            continue

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        desc = np.zeros(n_bits, dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(pattern):
            xr1 = int(x1 * cos_t - y1 * sin_t)
            yr1 = int(x1 * sin_t + y1 * cos_t)
            xr2 = int(x2 * cos_t - y2 * sin_t)
            yr2 = int(x2 * sin_t + y2 * cos_t)

            final_y1, final_x1 = y + yr1, x + xr1
            final_y2, final_x2 = y + yr2, x + xr2

            if (0 <= final_y1 < h and 0 <= final_x1 < w and
                0 <= final_y2 < h and 0 <= final_x2 < w):
                p1 = image[final_y1, final_x1]
                p2 = image[final_y2, final_x2]
                desc[i] = 1 if p1 < p2 else 0
        descriptors.append(desc)

    return np.array(descriptors, dtype=np.uint8)

def orb_feature_extraction(image):
    image = cv2.GaussianBlur(image, (5, 5), 1.2)
    keypoints = detect_keypoints(image)
    orientations = compute_orientation(image, keypoints)
    descriptors = generate_brief_descriptors(image, keypoints, orientations)
    return keypoints, descriptors

if __name__ == "__main__":
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb_feature_extraction(image)
    print(f"Extracted {len(keypoints)} keypoints")
    print(f" Descriptor shape: {descriptors.shape}")
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for (x, y) in keypoints:
        cv2.circle(output, (x, y), 2, (0, 255, 0), 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(output)
    plt.title(f"ORB-like Keypoints (Detected: {len(keypoints)})")
    plt.axis("off")
    plt.show()
