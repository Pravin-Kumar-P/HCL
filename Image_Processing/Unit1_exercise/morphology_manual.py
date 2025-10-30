import cv2
import numpy as np
import matplotlib.pyplot as plt
def pad_image(image, struct_elem):
    height, width = image.shape[:2]
    kernel_height, kernel_width = struct_elem.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded_image = np.zeros(
        (height + 2 * pad_h, width + 2 * pad_w),
        dtype=image.dtype
    )
    padded_image[pad_h:-pad_h, pad_w:-pad_w] = image
    return padded_image

def erode(image, struct_elem):
    padded = pad_image(image, struct_elem)
    height, width = image.shape
    kernel_height, kernel_width = struct_elem.shape
    eroded_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            window = padded[y:y + kernel_height, x:x + kernel_width]
            eroded_image[y, x] = np.min(window[struct_elem == 1])
    return eroded_image

def dilate(image, struct_elem):
    padded = pad_image(image, struct_elem)
    height, width = image.shape
    kernel_height, kernel_width = struct_elem.shape
    dilated_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            window = padded[y:y + kernel_height, x:x + kernel_width]
            dilated_image[y, x] = np.max(window[struct_elem == 1])
    return dilated_image

def opening(image, struct_elem):
    return dilate(erode(image, struct_elem), struct_elem)

def closing(image, struct_elem):
    return erode(dilate(image, struct_elem), struct_elem)

def create_square_struct_elem(size=3):
    return np.ones((size, size), dtype=np.uint8)

def main():
    image = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
    struct_elem = create_square_struct_elem(7)
    opened_image = opening(image, struct_elem)
    closed_image = closing(opened_image, struct_elem)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(opened_image, cmap='gray')
    plt.title("Opening")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(closed_image, cmap='gray')
    plt.title("Closing")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
