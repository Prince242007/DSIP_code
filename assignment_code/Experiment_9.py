import cv2
import numpy as np 
import matplotlib.pyplot as plt

image_path = 'bheem.jpg'
image = cv2.imread(image_path, 1)

plt.imshow(image,cmap="gray")
plt.show()  
# print(image)
print(image.shape)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imwrite(r"D:/college sem 5 pdf/Study/dsip/lecture_code/assignment_code/output_image.jpg",image)

plt.imshow(image_rgb)
plt.show()  

r,g,b = cv2.split(image)
fig, ax=plt.subplots(1,4,figsize=(16,4))
ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(r,cmap= 'Reds')
ax[1].set_title("Red")
ax[2].imshow(g,cmap= 'Greens')
ax[2].set_title("Green")
ax[3].imshow(b,cmap= 'Blues')
ax[3].set_title("Blue")

for a in ax : a.axis('on')
plt.show()



crop_image=image[117:310, 238:360]
plt.imshow(crop_image)
plt.show()

resize_image=cv2.resize(image,(300,300))
# plt.figure(figsize=(10,4))
plt.imshow(resize_image)
plt.show()