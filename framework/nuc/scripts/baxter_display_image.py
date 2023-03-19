#!/usr/bin/python2

import os
import sys
import argparse

import rospy

import cv2
import cv_bridge

from sensor_msgs.msg import (
    Image,
)


def send_image(path):
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)


def main():
    rospy.init_node('rsdk_xdisplay_image', anonymous=True)
    send_image("/home/installer/1.jpg")
    return 0

if __name__ == '__main__':
    sys.exit(main())
