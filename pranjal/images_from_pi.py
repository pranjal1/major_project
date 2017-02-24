#!/usr/bin/env python
#password is not needed as we are using private/public key for file transferring
try:
	import sys
	import os
except:
	sys.exit(1)

try:
	os.system("scp -r /home/pi/b_pthr.jpg pranjal@192.168.0.110:/home/pranjal/Desktop/images_pi")		
except:
	print "oops error"
	sys.exit(1)
