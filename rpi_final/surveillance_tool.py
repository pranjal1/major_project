#!/usr/bin/env python
import sys
import os
import time
import picamera

try:  
	import gi
	gi.require_version('Gtk', '3.0')
	from gi.repository import Gtk
except:  
	sys.exit(1)  


class loading_help_page:

	wTree = None
        
	


	def __init__( self ):
		self.gladefile = "help_page.glade" 
        	self.glade = Gtk.Builder()
        	self.glade.add_from_file(self.gladefile)
        	self.glade.connect_signals(self)
        	self.abc=self.glade.get_object("window1")
		self.abc.show_all()
		self.entryForText = self.glade.get_object("textview1")
		with open('help', 'r') as myfile:
    			data=myfile.read()
		self.entryForText.get_buffer().set_text(data)
		
		
	def quit_clicked(self, widget):
		self.abc.hide()
	
	


class raspberry_viewer:

	wTree = None

	def __init__( self ):
		self.gladefile = "./rpi_final_design.glade" 
        	self.glade = Gtk.Builder()
        	self.glade.add_from_file(self.gladefile)
        	self.glade.connect_signals(self)
        	self.window_main = self.glade.get_object("window1")
		self.window_main.resize(640,480)
		self.window_main.show_all()
		self.glade.get_object("label2").set_text("Start a session by clicking to start surveillance")
		self.mainImage = self.glade.get_object("image1")
      		self.mainImage.set_from_file("./apple.jpg")
   		self.mainImage.show()
	

	def quit_clicked(self, widget):
		sys.exit(0)

	def start_clicked(self,widget):
                self.glade.get_object("label2").set_text("Image capture and transmission")
                #image capture
                FRAMES = 5
                TIMEBETWEEN = 6

                frameCount = 0
                '''
                camera = picamera.PiCamera()
                camera.resolution = (640,480)

                while frameCount < FRAMES:
                    imageNumber = str(frameCount).zfill(7)
                    try:
                        camera.capture('/home/pi/Desktop/images_from_camera/image%s.jpg'%(imageNumber))
                    except:
                        print "Error in image capture!"
                        sys.exit(1)

                    frameCount += 1
                    time.sleep(TIMEBETWEEN - 6) #Takes roughly 6 seconds to take a picture
                '''

                #transferring captured images from raspberry pi to server(my pc in this case)
                number_of_images = FRAMES #change to frames!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                num_count =0
                WAIT_FOR_SCP_TRANSMIT = 6
                while num_count < number_of_images:
                        imageNumber = str(num_count).zfill(7)
                        try:
                                self.mainImage = self.glade.get_object("image1")
                                self.mainImage.set_from_file("/home/pi/Desktop/images_from_camera/image%s.jpg"%(imageNumber))
                                self.mainImage.show()
                                os.system("scp vvv -r /home/pi/Desktop/images_from_camera/image%s.jpg pranjal@192.168.0.110:/home/pranjal/Desktop/images_pi"%(imageNumber))
                                time.sleep(WAIT_FOR_SCP_TRANSMIT - 3) #Taking roughly 3 seconds to transmit image
                                
                        except Exception as abc:
                                print abc
                                sys.exit(1)
                        num_count+=1


		
	def stop_clicked(self,widget):
		print "abc"

	def help_clicked(self,widget):
		open_help = loading_help_page()

	def gtk_main_quit(self, widget):
		sys.exit(0)
		

letsdothis = raspberry_viewer()
Gtk.main()
