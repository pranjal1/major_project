import cv2
import numpy
import math

pi=3.1416
x=0
y=0

img = cv2.imread('ir.png',0)
'''
rows,cols= img.shape
 
M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
img = cv2.warpAffine(img,M,(cols,rows))
'''

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#center of the contours or estimated image points of the world coordinates
coord = []
coord1 = []

i=len(contours)-1
while i>=0:
	cnt = contours[i]
	M = cv2.moments(cnt)
	x = M['m10']/M['m00']
	y = M['m01']/M['m00']
	coord.append([int(x),int(y)])
	coord1.append([int(x*math.cos(pi/18)-y*math.sin(pi/18)),int(x*math.sin(pi/18)+y*math.cos(pi/18))])
	i-=1

#coordinate numbering part


coord1 = sorted(coord1)

font = cv2.FONT_HERSHEY_SIMPLEX
i=0
dummy = cv2.imread("ir.png")
while i<len(contours):
	temp = coord[i]
	cv2.putText(dummy,str(i),(int(temp[0]*math.cos(-pi/18)-temp[1]*math.sin(-pi/18)),int(temp[0]*math.sin(-pi/18)+temp[1]*math.cos(-pi/18))), font,0.5,(255,255,0),1,cv2.CV_AA)
	i+=1

cv2.imwrite("resulting_img.jpg",dummy)
cv2.imshow('numbering',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()
