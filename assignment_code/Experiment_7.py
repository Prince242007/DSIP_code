import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read image directly in grayscale
src_image = cv2.imread('oggy.jpg', cv2.IMREAD_GRAYSCALE)
# Check if image is loaded
if src_image is None:
    print("Image not found!")
    exit()
# ------------------------
# 1. Image Negation
# ------------------------
negative_image = 255 - src_image
# ------------------------
# 2. Thresholding
# ------------------------
_, thresholded_image = cv2.threshold(src_image, 128, 255, cv2.THRESH_BINARY)
# ------------------------
# 3. Gamma Correction
# ------------------------
gamma = 2.0
# Normalize image to range [0,1]
normalized_image = src_image / 255.0
# Apply gamma correction
gamma_corrected_image = np.power(normalized_image, 1/gamma)
# Convert back to range [0,255]
gamma_corrected_image = np.uint8(gamma_corrected_image * 255)
# ------------------------
# Display Images
# ------------------------
plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(negative_image, cmap='gray')
plt.title("Negative")
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Threshold")
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title("Gamma")
plt.axis('off')
plt.show()