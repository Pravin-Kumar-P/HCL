import cv2
import numpy as np
import matplotlib.pyplot as plt

img1_path = 'window.png'  
img2_path = 'image.png' 

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()  
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(img1, None) 
img_with_sift = cv2.drawKeypoints(img1, keypoints_sift1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

orb = cv2.ORB_create(nfeatures=200)  
keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(img1, None) 
img_with_orb = cv2.drawKeypoints(img1, keypoints_orb1, None, color=(0, 255, 0), flags=0)

keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(img2, None) 
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
matches = bf.match(descriptors_sift1, descriptors_sift2)  
matches = sorted(matches, key=lambda x: x.distance)[:10] 
match_image = cv2.drawMatches(img1, keypoints_sift1, img2, keypoints_sift2, matches, None, flags=2)

plt.figure(figsize=(15, 5))
plt.subplot(131)  
plt.imshow(img_with_sift, cmap='gray')
plt.title(f'SIFT Keypoints on Logo\n(Features: {len(keypoints_sift1)})')
plt.axis('off')

plt.subplot(132) 
plt.imshow(img_with_orb, cmap='gray')
plt.title(f'ORB Keypoints on Logo\n(Features: {len(keypoints_orb1)})')
plt.axis('off')

plt.subplot(133)  
plt.imshow(match_image)
plt.title(f'SIFT Matches\n(Matches: {len(matches)})')
plt.axis('off')

plt.tight_layout()
plt.show()
