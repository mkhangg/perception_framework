#!/usr/bin/env python

# Import libraries
import os
import rospy

from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

br = CvBridge()
import sys


old_time = time.time()
num_frame = 0
avg_fps = 0.0
def cb_cam(data):
    global old_time, num_frame, avg_fps
    new_time = time.time()
    num_frame += 1
    fps = 1/(new_time - old_time)
    avg_fps = (fps + avg_fps*(num_frame-1))/num_frame
    print("Left Cam: fps = %2.2f, avg_fps = %2.2f" % (fps, avg_fps))
    old_time = new_time
    frame = br.imgmsg_to_cv2(data, "bgr8")
    cv2.imshow(f"Baxter {sys.argv[1]} Cam", frame)
    cv2.waitKey(1)


# right_old_time = time.time()
# right_num_frame = 0
# right_avg_fps = 0.0
# def cb_right_cam(data):
#     global old_time, right_num_frame, right_avg_fps
#     new_time = time.time()
#     right_num_frame += 1
#     fps = 1/(new_time - right_old_time)
#     right_avg_fps = (fps + right_avg_fps*(right_num_frame-1))/right_num_frame
#     print("right Cam: fps = %2.2f, avg_fps = %2.2f" % (fps, right_avg_fps))
#     old_time = new_time
#     frame = br.imgmsg_to_cv2(data, "bgr8")
#     cv2.imshow("Baxter right Cam", frame)
#     cv2.waitKey(1)

def listener(cam):
    rospy.init_node(f'baxter_cam_{cam}_detector', anonymous=True)
    rospy.Subscriber(f'cameras/{cam}_hand_camera/image', Image, cb_cam)
    #rospy.Subscriber('cameras/right_hand_camera/image', Image, cb_right_cam)
    # spin() #simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener(sys.argv[1])
    
