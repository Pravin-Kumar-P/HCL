import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_gaussian_pyramid(image, num_octaves=4, scales_per_octave=5, sigma=1.6):
    k = 2 ** (1 / scales_per_octave)
    pyramid = []
    for o in range(num_octaves):
        octave = [image]
        for s in range(1, scales_per_octave):
            sigma_prev = sigma * (k ** (s - 1))
            sigma_curr = sigma * (k ** s)
            sigma_diff = np.sqrt(sigma_curr ** 2 - sigma_prev ** 2)
            blurred = cv2.GaussianBlur(octave[-1], (5, 5), sigmaX=sigma_diff, sigmaY=sigma_diff)
            octave.append(blurred)
        pyramid.append(octave)
        image = cv2.resize(octave[-1], (octave[-1].shape[1] // 2, octave[-1].shape[0] // 2))
    return pyramid

def build_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog = []
        for i in range(1, len(octave)):
            dog.append(octave[i] - octave[i - 1])
        dog_pyramid.append(dog)
    return dog_pyramid

def detect_keypoints(dog_pyramid, threshold=10):  # <-- raised threshold
    keypoints = []
    for o, octave in enumerate(dog_pyramid):
        for i in range(1, len(octave) - 1):
            prev_img, curr_img, next_img = octave[i - 1], octave[i], octave[i + 1]
            for x in range(1, curr_img.shape[0] - 1):
                for y in range(1, curr_img.shape[1] - 1):
                    val = curr_img[x, y]
                    patch = np.concatenate([
                        prev_img[x - 1:x + 2, y - 1:y + 2].flatten(),
                        curr_img[x - 1:x + 2, y - 1:y + 2].flatten(),
                        next_img[x - 1:x + 2, y - 1:y + 2].flatten()
                    ])
                    if (val == patch.max() or val == patch.min()) and abs(val) > threshold:
                        keypoints.append((x * (2 ** o), y * (2 ** o), o))
    return keypoints

def compute_orientation(image, keypoints):
    oriented_keypoints = []
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 360

    for x, y, _ in keypoints:
        x, y = int(x), int(y)
        if 8 < x < image.shape[0] - 8 and 8 < y < image.shape[1] - 8:
            region_mag = magnitude[x - 8:x + 8, y - 8:y + 8]
            region_ang = angle[x - 8:x + 8, y - 8:y + 8]
            hist, _ = np.histogram(region_ang, bins=36, range=(0, 360), weights=region_mag)
            max_val = hist.max()
            peaks = np.where(hist > 0.8 * max_val)[0]
            for peak in peaks:
                dominant_angle = peak * 10
                oriented_keypoints.append((x, y, dominant_angle))
    return oriented_keypoints

def compute_descriptors(image, keypoints):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 360) % 360

    descriptors = []
    for x, y, orientation in keypoints:
        x, y = int(x), int(y)
        if x < 8 or y < 8 or x + 8 >= image.shape[0] or y + 8 >= image.shape[1]:
            continue
        patch_mag = magnitude[x - 8:x + 8, y - 8:y + 8]
        patch_ang = (angle[x - 8:x + 8, y - 8:y + 8] - orientation) % 360
        desc = []
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                cell_mag = patch_mag[i:i + 4, j:j + 4]
                cell_ang = patch_ang[i:i + 4, j:j + 4]
                hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                desc.extend(hist)
        desc = np.array(desc)
        descriptors.append(desc)
    return np.array(descriptors)

def sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gaussian_pyramid = build_gaussian_pyramid(img)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)
    keypoints = detect_keypoints(dog_pyramid, threshold=10)
    oriented_keypoints = compute_orientation(img, keypoints)
    descriptors = compute_descriptors(img, oriented_keypoints)

    print(f"Total keypoints detected: {len(oriented_keypoints)}")
    return img, oriented_keypoints, descriptors

if __name__ == "__main__":
    image_path = "image.png"
    img_gray, kps, desc = sift_features(image_path)

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    for (x, y, ang) in kps:
        cv2.circle(img_color, (int(y), int(x)), 2, (0, 255, 0), 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_color)
    plt.title("SIFT Keypoints (Manual Implementation)")
    plt.axis("off")
    plt.show()
