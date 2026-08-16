# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Read image
# image = cv2.imread(r"open_ended_7\Moon.jpg")

# if image is None:
#     print("Error: Image not found")
#     exit()

# # Convert BGR to RGB for displaying with Matplotlib
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Convert to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Display original image
# plt.figure(figsize=(8, 5))
# plt.imshow(rgb_image)
# plt.title("Original Image")
# plt.axis("off")
# plt.show()

# # Display grayscale image
# plt.figure(figsize=(8, 5))
# plt.imshow(gray_image, cmap="gray")
# plt.title("Grayscale Image")
# plt.axis("off")
# plt.show()


# # Convert BGR image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Display grayscale image
# plt.figure(figsize=(8, 5))
# plt.imshow(gray_image, cmap="gray")
# plt.title("Original Grayscale Image")
# plt.axis("off")
# plt.show()


# print("Minimum intensity:", np.min(gray_image))
# print("Maximum intensity:", np.max(gray_image))

# # Contrast Stretching

# r_min = np.min(gray_image)
# r_max = np.max(gray_image)

# stretched = ((gray_image - r_min) / (r_max - r_min)) * 255
# stretched = stretched.astype(np.uint8)

# # Display enhanced image
# plt.figure(figsize=(8, 5))
# plt.imshow(stretched, cmap="gray")
# plt.title("Enhanced Image - Contrast Stretching")
# plt.axis("off")
# plt.show()

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(gray_image, cmap="gray")
# plt.title("Before Contrast Stretching")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(stretched, cmap="gray")
# plt.title("After Contrast Stretching")
# plt.axis("off")

# plt.show()





#///////////////////////////////////////////////////////////////////////////////////////////
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1. Read Image
# --------------------------------------------------

image = cv2.imread(r"open_ended_7\oggy.jpg")

if image is None:
    print("Error: Image not found")
    exit()

print("Image loaded successfully!")


# --------------------------------------------------
# 2. Convert BGR to RGB
# --------------------------------------------------

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 3. Convert Color Image to Grayscale
# --------------------------------------------------

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------
# 4. Display Original Color Image
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.title("Original Color Image - Oggy")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 5. Display Grayscale Image
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 6. Gamma Transformation
# --------------------------------------------------

gamma = 0.5

# Normalize pixel values from 0-255 to 0-1
normalized = gray_image / 255.0

# Apply gamma transformation
gamma_image = np.power(normalized, gamma)

# Convert back to 0-255
gamma_image = np.uint8(gamma_image * 255)


# --------------------------------------------------
# 7. Display Enhanced Image
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.imshow(gamma_image, cmap="gray")
plt.title("Enhanced Image - Gamma Transformation")
plt.axis("off")
plt.show()


# --------------------------------------------------
# 8. Before and After Comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Before Gamma Transformation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gamma_image, cmap="gray")
plt.title("After Gamma Transformation")
plt.axis("off")

plt.show()


# --------------------------------------------------
# 9. Histogram Before and After
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), bins=256, range=[0, 256])
plt.title("Histogram Before Enhancement")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.subplot(1, 2, 2)
plt.hist(gamma_image.ravel(), bins=256, range=[0, 256])
plt.title("Histogram After Gamma Transformation")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")

plt.show()


# --------------------------------------------------
# 10. Display Minimum and Maximum Intensities
# --------------------------------------------------

print("Before Enhancement:")
print("Minimum intensity:", np.min(gray_image))
print("Maximum intensity:", np.max(gray_image))

print("\nAfter Gamma Transformation:")
print("Minimum intensity:", np.min(gamma_image))
print("Maximum intensity:", np.max(gamma_image))