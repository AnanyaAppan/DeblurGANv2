import cv2 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
    
# Using cv2.imread() method 
img1 = cv2.imread('real.png') 
img2 = cv2.imread('fake.png')

print("real")
print(img1[:5,:5,2])
print("fake")
print(img2[:5,:5,2])

# print("real")
# print(img1)
# print("fake")
# print(img2)
  
#Displaying the image 
c = plt.imshow(img1[:150,:150,:])
plt.show()
d = plt.imshow(img2[:150,:150,:]) 
plt.show()