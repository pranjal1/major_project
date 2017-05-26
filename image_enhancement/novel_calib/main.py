import cv2
import numpy as np

img = cv2.imread("ir.png",0)
#hardcoded thresholding worked a little well for now
#ret,th2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#edges = cv2.Canny(th2,100,200)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.imshow('ir_image',img)
cv2.waitKey(0)

#center of the contours or estimated image points of the world coordinates
cx = [None]*len(contours)
cy = [None]*len(contours)


i=len(contours)-1
while i>=0:
	cnt = contours[i]
	M = cv2.moments(cnt)
	cx[i] = int(M['m10']/M['m00'])
	cy[i] = int(M['m01']/M['m00'])
	i-=1


for ind, cont in enumerate(contours):
    elps = cv2.fitEllipse(cont)
    cv2.ellipse(th2,elps,(255,0,0),2)


cv2.imshow('ellipses',th2)
cv2.waitKey(0)

i=0
dummy = cv2.imread("ir.png")
while i<len(contours):
	cv2.circle(dummy,(cx[i],cy[i]), 2, (0,0,255), -1)
	i+=1

cv2.imwrite('result.jpg',dummy)
cv2.imshow('with centre of mass',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()


#actual points (taking Z=0)
rx = [None]*len(contours)
ry = [None]*len(contours)

i=0
while i<int((len(contours))**0.5):
	j=0
	while j<int((len(contours))**0.5):	
		rx[i] = i
		ry[j] = j 
		j+=1
	i+=1

i=int((len(contours))**0.5)-1
count=0
while i>=0:
	j=int((len(contours))**0.5)-1
	while j>=0:	
		print "("+str(cx[count])+","+str(cy[count])+")"+"======>"+"("+str(rx[i])+","+str(ry[j])+")"
		j-=1
		count+=1
	print "-------------------------------"
	i-=1















