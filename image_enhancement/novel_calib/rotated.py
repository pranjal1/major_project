import cv2
import numpy as np

img = cv2.imread('ir.png',0)


ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.imshow('ir_image',img)
cv2.waitKey(0)

#center of the contours or estimated image points of the world coordinates
coord = []

i=len(contours)-1
while i>=0:
	cnt = contours[i]
	M = cv2.moments(cnt)
	coord.append([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
	i-=1

#coord = sorted(coord)
print coord

for ind, cont in enumerate(contours):
    elps = cv2.fitEllipse(cont)
    cv2.ellipse(th2,elps,(255,0,0),2)


cv2.imshow('ellipses',th2)
cv2.waitKey(0)

i=0
dummy = cv2.imread("ir.png")
while i<len(contours):
	temp = coord[i]
	cv2.circle(dummy,(temp[0],temp[1]), 2, (0,0,255), -1)
	i+=1

cv2.imshow('with centre of mass',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()




# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, corners = cv2.findChessboardCorners(th2, (9,9),None)








