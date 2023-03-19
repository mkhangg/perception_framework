#! /usr/bin/python

# Import libraries
import os
import cv2
import time
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Instantiate CvBridge
bridge = CvBridge()

t_old = time.time()
index = 0 

def image_callback(msg):
    global t_old
    print("Received an image!")
    t_new = time.time()
    print("FPS = %2.2f" % (1/(t_new - t_old)))
    t_old = t_new
    global index
    index = index  + 1

    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "16UC1")
        # flatten_depth_img = np.frombuffer(msg.data, dtype=np.uint16)  # shape =(width*height,)
        # cv2_img = flatten_depth_img.reshape(msg.height, msg.width) # shape =(width, height)
    except CvBridgeError, error:
        print(error)
    else:
        cv2.imshow('image', cv2_img)
        # cv2.imwrite("demo_" + str(index) + ".jpg", cv2_img)
        cv2.waitKey(1)

def image_listener():
    rospy.init_node('image_listener', anonymous=True)
    rospy.Subscriber("/d435_camera/depth/image_rect_raw", Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    image_listener()