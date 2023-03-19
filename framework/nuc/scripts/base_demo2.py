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

import tf

import geometry_msgs
import math

def rad2deg(rad):
    return ((rad/math.pi)*180.0)

b_run = True
def signal_handler(sig, frame):
    global b_run
    print('You pressed Ctrl+C!')
    b_run = False
    #sys.exit(0)

old_time = time.time()
num_frame = 0
avg_fps = 0.0

x=list()
yl=list()
yr=list()
ym = list()
i = 0
start = 0
chunk = 100
l = 0

x3d =list()
y3d =list()
z3d =list()
        
def callback_scan(scan_data): # 10Hz
    global old_time, num_frame, avg_fps, i, x, yl, yr, ym
    new_time = time.time()
    num_frame += 1
    fps = 1/(new_time - old_time)
    avg_fps = (fps + avg_fps*(num_frame-1))/num_frame
    old_time = new_time
    #print("Base /scan: fps = %2.2f, avg_fps = %2.2f" % (fps, avg_fps))
    #print("scan_data", scan_data)
    n = len(scan_data.ranges)
    num_data_per_degree = n/360.0
    print('num_data_per_degree = ', num_data_per_degree)
    fov = 90    #Field of View Angle Forward Direction
    mid = int(0.5*n)
    left = int(mid + fov*num_data_per_degree + 1)
    right = int(mid - fov*num_data_per_degree - 1)
    print(left, mid, right)
    
    i += 1
    delta_angle = 2
    num_data = int(num_data_per_degree*delta_angle) + 1  #number_data_per_delta_angle
    print('num_data = ', num_data)
    x.append(i)
    yl.append(max(scan_data.ranges[left-num_data:left+num_data]))
    ym.append(max(scan_data.ranges[mid-num_data:mid+num_data]))
    yr.append(max(scan_data.ranges[right-num_data:right+num_data]))
    

    x3d.append(-90)
    y3d.append(max(scan_data.ranges[left-num_data:left+num_data]))
    z3d.append(i)

    x3d.append(0)
    y3d.append(max(scan_data.ranges[mid-num_data:mid+num_data]))
    z3d.append(i)

    x3d.append(90)
    y3d.append(max(scan_data.ranges[right-num_data:right+num_data]))
    z3d.append(i)
    
    global l, start, chunk
    l = len(x)
    if l > chunk:
        start = l - chunk


x_rot = list()
y_rot = list()
i_rot = 0
rot = 0
rot_start = 0
rot_length = 0
rot_chunck = 20000

rot_old_time = time.time()
rot_num_frame = 0
rot_avg_fps = 0.0



def callback_odom(odom_data): #250Hz
    global rot_old_time, rot_num_frame, rot_avg_fps
    rot_new_time = time.time()
    rot_num_frame += 1
    fps = 1/(rot_new_time - rot_old_time)
    rot_old_time = rot_new_time
    rot_avg_fps = (fps + rot_avg_fps*(rot_num_frame-1))/rot_num_frame
    #print("Base /odom: fps = %2.2f, avg_fps = %2.2f" % (fps, rot_avg_fps))

    global i_rot, rot, x_rot, y_rot, rot_start, rot_length
    #print(odom_data)
    quat = odom_data.pose.pose.orientation
    #print([quat.x, quat.y, quat.z, quat.w])
    euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    #print(rad2deg(euler[0]), rad2deg(euler[1]), rad2deg(euler[2]))
    i_rot += 1
    rot = rad2deg(euler[2])
    x_rot.append(i_rot)
    y_rot.append(rot)
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
    fig, axes = plt.subplots(1, 2,  figsize=(16, 4))

    #fig1 = plt.figure()
    global b_run, x, yl, yr, ym, start, l
    global rot, x_rot, y_rot, rot_start, rot_length
    while b_run:
        
        # ylim = 10

        # axes[0].clear()
        # axes[0].plot(x[start:l], yl[start:l])
        # axes[0].set_title('LEFT')
        # axes[0].set_ylim(ymin=0, ymax=ylim)

        # axes[1].clear()
        # axes[1].plot(x[start:l], ym[start:l])
        # axes[1].set_title('MIDDLE')
        # axes[1].set_ylim(ymin=0, ymax=ylim)

        #axes[2].clear()
        # axes[2].plot(x[start:l], yr[start:l])
        # axes[2].set_title('RIGHT')
        # axes[2].set_ylim(ymin=0, ymax=ylim)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.clear()
        ax.scatter(z3d, z3d, x3d)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        axes[1].clear()
        axes[1].plot(x_rot[rot_start:rot_length], y_rot[rot_start:rot_length])
        axes[1].set_title('Z ROTATION')
        axes[1].set_ylim(ymin=-180, ymax=180)
        plt.show()
        plt.pause(0.5) #Note this correction

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    plt.ion() ## Note this correction
    listener()
    