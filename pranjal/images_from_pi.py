#!/usr/bin/env python
#password is not needed as we are using private/public key for file transferring
import sys
import os
import picamera
import time


FRAMES = 1000
TIMEBETWEEN = 6

frameCount = 0

while frameCount < FRAMES:
    imageNumber = str(frameCount).zfill(7)
    try:
    	os.system("raspistill -w 640 -h 480 -hf -o /home/pi/Desktop/images_from_camera/image%s.jpg"%(imageNumber))
    except:
    	print "Error in image capture!"
    	sys.exit(1)

    frameCount += 1
    time.sleep(TIMEBETWEEN - 6) #Takes roughly 6 seconds to take a picture


#transferring files from raspberry pi to server(my pc in this case) using scp
number_of_images = FRAMES
num_count =0
WAIT_FOR_SCP_TRANSMIT = 6
while num_count < number_of_images:
	imageNumber = str(num_count).zfill(7)
	try:
		os.system("scp -r /home/pi/Desktop/images_from_camera/image%s.jpg pranjal@192.168.0.110:/home/pranjal/Desktop/images_pi"%(imageNumber))
		time.sleep(WAIT_FOR_SCP_TRANSMIT - 3) #Taking roughly 3 seconds to transmit image		
	except:
		print "Error in scp file transmission"
		sys.exit(1)
		num_count+=1

