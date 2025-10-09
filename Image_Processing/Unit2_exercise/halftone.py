import cv2
import numpy as np
import matplotlib.pyplot as plt

def ordered_dithering(img):
    bayer_matrix = (1/16) * np.array([[0, 8, 2, 10],
                                      [12, 4, 14, 6],
                                      [3, 11, 1, 9],
                                      [15, 7, 13, 5]])
    h, w = img.shape
    tile_h = int(np.ceil(h / 4))
    tile_w = int(np.ceil(w / 4))
    threshold_map = np.tile(bayer_matrix, (tile_h, tile_w))
    threshold_map = threshold_map[:h, :w] * 255
    halftoned = (img > threshold_map).astype(np.uint8) * 255
    return halftoned


def error_diffusion(img):
    """Apply Floyd–Steinberg error diffusion halftoning"""
    img = img.astype(float)
    h, w = img.shape
    halftoned = np.copy(img)
    for y in range(h):
        for x in range(w):
            old_pixel = halftoned[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            halftoned[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x+1 < w:
                halftoned[y, x+1] += quant_error * 7/16
            if y+1 < h and x > 0:
                halftoned[y+1, x-1] += quant_error * 3/16
            if y+1 < h:
                halftoned[y+1, x] += quant_error * 5/16
            if y+1 < h and x+1 < w:
                halftoned[y+1, x+1] += quant_error * 1/16
    halftoned = np.clip(halftoned, 0, 255).astype(np.uint8)
    return halftoned


if __name__ == "__main__":
    # Load grayscale image
    img_path = "image.png" 
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Apply both halftoning techniques
    ordered = ordered_dithering(image)
    floyd = error_diffusion(image)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ordered Dithering (Bayer)")
    plt.imshow(ordered, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Error Diffusion (Floyd–Steinberg)")
    plt.imshow(floyd, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
