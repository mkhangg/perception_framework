#!/usr/bin/env python

# Import libraries
import os
import rospy

from std_msgs.msg import String
from nav_msgs.msg import Odometry

fileName = "t265_data.txt"

import time
t_old = time.time()

# Callback function getting camera data 
def callback(data):
    global t_old    
    t_new = time.time()
    print("FPS = %2.2f" % (1/(t_new - t_old)))
    t_old = t_new

    # Get position [x, y, z] and orientation
    rospy.loginfo(">> Position [p_x, p_y, p_z]")
    rospy.loginfo("p_x = %s", data.pose.pose.position.x)
    rospy.loginfo("p_y = %s", data.pose.pose.position.y)
    rospy.loginfo("p_z = %s", data.pose.pose.position.z)

    rospy.loginfo(">> Orientation [q_x, q_y, q_z, q_w]")
    rospy.loginfo("q_x = %s", data.pose.pose.orientation.x)
    rospy.loginfo("q_y = %s", data.pose.pose.orientation.y)
    rospy.loginfo("q_z = %s", data.pose.pose.orientation.z)
    rospy.loginfo("q_w = %s", data.pose.pose.orientation.w)

    # Get linear and angular acceleration
    rospy.loginfo(">> Linear Acceleration [l_x, l_y, l_z]")
    rospy.loginfo("l_x = %s", data.twist.twist.linear.x)
    rospy.loginfo("l_y = %s", data.twist.twist.linear.y)
    rospy.loginfo("l_z = %s", data.twist.twist.linear.z)

    rospy.loginfo(">> Angular Acceleration [a_x, a_y, a_z]")
    rospy.loginfo("a_x = %s", data.twist.twist.angular.x)
    rospy.loginfo("a_y = %s", data.twist.twist.angular.y)
    rospy.loginfo("a_z = %s", data.twist.twist.angular.z)

    rospy.loginfo("=======================================")

    # Print data to file
    '''
    with open(fileName, "a") as fw:
        fw.write("%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n" 
            %(data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z,
            data.pose.pose.orientation.x, data.pose.pose.orientation.y, 
            data.pose.pose.orientation.z, data.pose.pose.orientation.w,
            data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z,
            data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z))
    fw.close()
    '''

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/t265_camera//odom/sample', Odometry, callback)
    
    with open(fileName, "a") as fw:
        fw.write("p_x, p_y, p_z, o_x, o_y, o_z, o_w, l_x, l_y, l_z, a_x, a_y, a_z\n")
    # spin() #simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    # Delete file if exists
    if os.path.exists(fileName):
       os.remove(fileName)
    
    listener()
    
