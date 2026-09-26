import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# image = cv2.imread('bheem.jpg', 1)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb)
# plt.show()

# def apply_median_filter(image_rgb, kernel_size): 
# # Apply median filter to remove noise 
#     filtered_image = cv2.medianBlur(image_rgb, kernel_size) 
#     return filtered_image

# def apply_bilateral_filter(image_rgb, d, sigma_color, sigma_space): 
# # Apply bilateral filter to remove noise 
#     filtered_image = cv2.bilateralFilter(image_rgb, d, sigma_color, sigma_space) 
#     return filtered_image

# # Apply median filter 
# median_filtered_image = apply_median_filter(image_rgb, kernel_size=5) 
# # Apply bilateral filter 
# bilateral_filtered_image = apply_bilateral_filter(image_rgb, d=9, sigma_color=75, sigma_space=75) 

# plt.imshow(median_filtered_image)
# plt.show()
# plt.imshow(bilateral_filtered_image)
# plt.show()


# #----------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------
# from PIL import Image, ImageFilter
# img_123 = Image.open("bheem.jpg")
# min_filtered_img = img_123.filter(ImageFilter.MinFilter(size=3))

# # 3. Apply the Max Filter
# # size=3 creates a 3x3 pixel neighborhood
# max_filtered_img = img_123.filter(ImageFilter.MaxFilter(size=3))

# plt.imshow(min_filtered_img)
# plt.show()
# plt.imshow(max_filtered_img)
# plt.show()


image = cv2.imread('p.png', 1)
kernel = np.ones((3, 3), np.uint8)

# 3. Apply Max Filter using Dilation
# Replaces the center pixel with the maximum value in the 3x3 neighborhood
max_filtered = cv2.dilate(image, kernel)

# 4. Apply Min Filter using Erosion
# Replaces the center pixel with the minimum value in the 3x3 neighborhood
min_filtered = cv2.erode(image, kernel)
plt.imshow(image)
plt.show()
plt.imshow(min_filtered)
plt.show()
plt.imshow(max_filtered)
plt.show()

