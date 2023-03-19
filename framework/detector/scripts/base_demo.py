#!/usr/bin/env python

# Import libraries
import os
import rospy

from datetime import datetime
from sensor_msgs.msg import LaserScan
from  nav_msgs.msg import Odometry
import time

import matplotlib.pyplot as plt

import signal
import sys
from geometry_msgs.msg import Twist

import tf

import geometry_msgs
import math


#global twist_output
#pub.publish(twist_output)

pose, prev_pose = 0.0, 0.0


def rad2deg(rad):
    return ((rad/math.pi)*180.0)

b_run = True
def signal_handler(sig, frame):
    global b_run
    print('You pressed Ctrl+C!')
    b_run = False

x=list()
yl=list()
yr=list()
ym = list()
i = 0
start = 0
chunk = 100
l = 0
        
def callback_scan(scan_data): # 10Hz
    #Run parameters
    speed = 0.25            #m/s
    check_forward_dist = 0.85 #Meter
    check_side_dist = 0.7    #Meter
    escape_range = 10 #Dregree
    fov = 45
    fov_center = 25
    delta_angle = 90    #Field of View Angle Forward Direction

    n = len(scan_data.ranges)
    num_data_per_degree = n/360.0
    #print('num_data_per_degree = %2.2f' %( num_data_per_degree))
    
    mid = int(0.5*n)
    left = int(mid + delta_angle*num_data_per_degree + 1)
    right = int(mid - delta_angle*num_data_per_degree - 1)
    #print(left, mid, right)
    global i
    i += 1    
    num_data = int(num_data_per_degree*fov) + 1  #number_data_per_delta_angle
    num_data_center = int(num_data_per_degree*fov_center) + 1  #number_data_per_delta_angle
    #print('num_data = ', num_data)
    x.append(i)
    yl_data = min([x for x in scan_data.ranges[left-num_data:left+num_data] if x > 0])
    yr_data = min([x for x in scan_data.ranges[right-num_data:right+num_data]  if x > 0])
    ym_data = min([x for x in scan_data.ranges[mid-num_data_center:mid+num_data_center] if x > 0])
    print(yl_data, ym_data, yr_data)
    yl.append(yl_data)
    yr.append(yr_data)
    ym.append(ym_data)
    global l, start, chunk
    l = len(x)
    if l > chunk:
        start = l - chunk

    global pub, pose, prev_pose    

    GET_DIR = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    FORWARD = 3

    state = GET_DIR

    def move(lin_x_vel, ang_z_vel):
        cmd_vel = Twist()
        cmd_vel.linear.x = lin_x_vel
        cmd_vel.angular.z = ang_z_vel
        #pub.publish(cmd_vel)

    if ym_data > check_forward_dist:
        if yl_data < check_side_dist:
            state = TURN_RIGHT
        elif yr_data < check_side_dist:
            state = TURN_LEFT
        else:
            state = FORWARD
    else:
        if yl_data < yr_data:
            state = TURN_RIGHT
        else:
            state = TURN_LEFT            

    b_in_range = True if math.fabs(prev_pose - pose) < escape_range else False
    if state == TURN_LEFT and b_in_range:
        print("TURN LEFT")
        move(0, speed*1.2)
    elif state == TURN_RIGHT and b_in_range:
        print("TURN RIGHT")
        move(0, -speed*1.2)
    elif state == FORWARD:
        print("GO FORWARD")
        move(speed, 0)
    else:
        print("GET DIRECTION")
    print("prev_pose = %2.2f, pose = %2.2f" % (prev_pose, pose))
    prev_pose = pose

#Rotation variable
x_rot = list()
y_rot = list()
i_rot = 0
rot = 0
rot_start = 0
rot_length = 0
rot_chunck = 20000

def callback_odom(odom_data): #250Hz
    global i_rot, rot, x_rot, y_rot, rot_start, rot_length, pose
    quat = odom_data.pose.pose.orientation
    euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    i_rot += 1
    pose = rad2deg(euler[2])
    x_rot.append(i_rot)
    y_rot.append(pose)
    rot_length = len(x_rot)
    if rot_length > rot_chunck:
        rot_start = rot_length - rot_chunck
        # x_rot = x_rot[rot_start:rot_length]
        # y_rot = y_rot[rot_start:rot_length]
        # rot_start = 0


def listener():
    rospy.init_node(f'node_scan', anonymous=True)
    rospy.Subscriber(f'/scan', LaserScan, callback_scan)
    rospy.Subscriber(f'/odom', Odometry, callback_odom)
    #rospy.spin()
    fig, axes = plt.subplots(1, 4,  figsize=(12.2, 2.6))

    global b_run, x, yl, yr, ym, start, l
    global rot, x_rot, y_rot, rot_start, rot_length
    while b_run:
        
        ylim = 10

        axes[0].clear()
        axes[0].plot(x[start:l], yl[start:l])
        axes[0].set_title('LEFT')
        axes[0].set_ylim(ymin=0, ymax=ylim)

        axes[1].clear()
        axes[1].plot(x[start:l], ym[start:l])
        axes[1].set_title('MIDDLE')
        axes[1].set_ylim(ymin=0, ymax=ylim)

        axes[2].clear()
        axes[2].plot(x[start:l], yr[start:l])
        axes[2].set_title('RIGHT')
        axes[2].set_ylim(ymin=0, ymax=ylim)
        

        axes[3].clear()
        axes[3].plot(x_rot[rot_start:rot_length], y_rot[rot_start:rot_length])
        axes[3].set_title('Z ROTATION')
        axes[3].set_ylim(ymin=-180, ymax=180)
        plt.show()
        plt.pause(0.5) #Note this correction

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    pub = rospy.Publisher('/mobility_base/cmd_vel', Twist, queue_size=1)
    plt.ion() ## Note this correction
    listener()
    