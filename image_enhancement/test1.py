import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image1.jpg',0)
equ = cv2.equalizeHist(img) #tried to see if histogram equalizing improved edge detecting efficiency and results are good. 

'''
#histogram visualizing

hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
'''

edges = cv2.Canny(equ,100,200) #edge is detected from histogram equalized image
dst = cv2.add(img,edges) #result is added to original image as the histogram equalized image has blown out whites

plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(dst,cmap='gray')
plt.title('overlap Image'), plt.xticks([]), plt.yticks([])
plt.show()

   
