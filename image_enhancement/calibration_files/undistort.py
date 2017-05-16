import numpy as np
import cv2
import glob
import sys
import pickle


#argument parsing part here
try:
	first_arg = sys.argv[1]
	second_arg = sys.argv[2]

except:
	print "arguments mismatch"

#file reading part

fh = open('./calibration_params/ret.txt', 'r') 
ret = float(fh.readline()) 
fh.close()


dist = np.load('./calibration_params/dist.txt')
mtx = np.load('./calibration_params/mtx.txt')

with open ('./calibration_params/rvecs.txt', 'rb') as fp:
    rvecs = pickle.load(fp)
fp.close()

with open ('./calibration_params/tvecs.txt', 'rb') as fp:
    tvecs = pickle.load(fp)
fp.close()


img = cv2.imread(first_arg)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(second_arg,dst)

cv2.destroyAllWindows()

