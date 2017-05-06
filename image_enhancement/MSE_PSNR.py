import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('net2.jpg',0)

equ = cv2.imread('net1.jpg',0)

try:
	data1 = np.asarray( img, dtype='uint8' )
	data2 = np.asarray( equ, dtype='uint8' )
except SystemError:
	data1 = np.asarray( img.getdata(), dtype='uint8' )
	data2 = np.asarray( equ.getdata(), dtype='uint8' )

data1 = data1.astype(np.float64)
data2 = data2.astype(np.float64)

height = np.size(data1, 0)
width = np.size(data1, 1)



row_count = height - 1
col_count = width - 1 
diff_sq = 0.0
sum_sq = 0.0
mse = 0.0
psnr = 0.0
diff = 0.0

while (row_count != 0):
	col_count = width - 1
	while (col_count != 0):
		diff = (data1[row_count,col_count] - data2[row_count,col_count])
		diff_sq = diff ** 2
		sum_sq += diff_sq
		col_count -= 1
	row_count -= 1

mse = sum_sq/(height*width)
			
print "MSE = ",mse

try:
	psnr = 20 * math.log10(255) - 10 * math.log10(mse)
	print "PSNR = ",psnr
except ValueError:
	print "psnr is infinite"




	



   
