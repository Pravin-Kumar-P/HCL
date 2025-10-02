import cv2
import numpy as np

img = cv2.imread("004.png", cv2.IMREAD_GRAYSCALE)

_, otsu = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

adaptive = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11, 2)

def add_label(image, text):
    labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    cv2.putText(labeled, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return labeled

img_l      = add_label(img, "Original")
otsu_l     = add_label(otsu, "Otsu Threshold")
adaptive_l = add_label(adaptive, "Adaptive Threshold")
combined = cv2.hconcat([img_l, otsu_l, adaptive_l])
scale = 0.7
combined_small = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cv2.imshow("Thresholding Comparison", combined_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
