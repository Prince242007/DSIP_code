import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("oggy.jpg",0)
kernel = np.ones((5,5), np.uint8)
_,thres_img = cv2.threshold(image,128,255,cv2.THRESH_BINARY)

# dilation
dilation = cv2.dilate(thres_img,kernel,iterations = 1)

# plt.plot(dilation)

plt.figure(figsize=(6,4))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.axis("off")
plt.imshow(image,cmap="gray")

plt.subplot(2,2,2)
plt.title("threshold Image")
plt.axis("off")
plt.imshow(thres_img,cmap="gray")

plt.subplot(2,2,3)
plt.title("Image after dilation")
plt.axis("off")
plt.imshow(dilation,cmap="gray")

boundry = thres_img - dilation
plt.subplot(2,2,4)
plt.title("boundry Image")
plt.axis("off")
plt.imshow(boundry,cmap="gray")

plt.tight_layout()


plt.show()