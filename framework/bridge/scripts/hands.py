#!/usr/bin/env python

# Example code for a script using the ReFlex Takktile hand
# Note: you must connect a hand by running "roslaunch reflex reflex_takktile.launch" before you can run this script


from math import pi, cos

import rospy
from std_srvs.srv import Empty

from reflex_msgs.msg import Command
from reflex_msgs.msg import PoseCommand
from reflex_msgs.msg import VelocityCommand
from reflex_msgs.msg import ForceCommand
from reflex_msgs.msg import Hand
from reflex_msgs.msg import FingerPressure
from reflex_msgs.srv import SetTactileThreshold, SetTactileThresholdRequest

import threading
import math

import sys #Use arguments
concurrency = 0
if sys.argv > 1:
    concurrency = sys.argv[1]

print(sys.argv)

def deg2rad(deg):
    return ((deg*math.pi)/180.0)

hand_state = Hand()

# TODO: update script for two hands (probably need a second hand object)

def move_thread(pos_pub_hand, name, points):
    for p in points:
        #print('Hand ', name, ' move to: ', p)
        pos_pub_hand.publish(p)
        rospy.sleep(0.5)

def main():
    ##################################################################################################################
    rospy.init_node('ExampleHandNode')

    # Hand 1

    print("Initiallizing Hand 1\n")

    # Services can automatically call hand calibration
    calibrate_fingers_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_fingers', Empty)
    calibrate_tactile_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_tactile', Empty)
    
    # Services can set tactile thresholds and enable tactile stops
    enable_tactile_stops_hand1 = rospy.ServiceProxy('/reflex_takktile/enable_tactile_stops', Empty)
    disable_tactile_stops_hand1 = rospy.ServiceProxy('/reflex_takktile/disable_tactile_stops', Empty)
    set_tactile_threshold_hand1 = rospy.ServiceProxy('/reflex_takktile/set_tactile_threshold', SetTactileThreshold)

    # This collection of publishers can be used to command the hand
    command_pub_hand1 = rospy.Publisher('/reflex_takktile/command', Command, queue_size=1)
    pos_pub_hand1 = rospy.Publisher('/reflex_takktile/command_position', PoseCommand, queue_size=1)
    vel_pub_hand1 = rospy.Publisher('/reflex_takktile/command_velocity', VelocityCommand, queue_size=1)
    force_pub_hand1 = rospy.Publisher('/reflex_takktile/command_motor_force', ForceCommand, queue_size=1)

    # Constantly capture the current hand state
    rospy.Subscriber('/reflex_takktile/hand_state', Hand, hand_state_cb)

    ##################################################################################################################

    # Hand 2

    print("Initiallizing Hand 2\n")

    # Services can automatically call hand calibration
    calibrate_fingers_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_fingers', Empty)
    calibrate_tactile_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_tactile', Empty)
    
    # Services can set tactile thresholds and enable tactile stops
    enable_tactile_stops_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/enable_tactile_stops', Empty)
    disable_tactile_stops_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/disable_tactile_stops', Empty)
    set_tactile_threshold_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/set_tactile_threshold', SetTactileThreshold)

    # This collection of publishers can be used to command the hand
    command_pub_hand2 = rospy.Publisher('/hand2/reflex_takktile/command', Command, queue_size=1)
    pos_pub_hand2 = rospy.Publisher('/hand2/reflex_takktile/command_position', PoseCommand, queue_size=1)
    vel_pub_hand2 = rospy.Publisher('/hand2/reflex_takktile/command_velocity', VelocityCommand, queue_size=1)
    force_pub_hand2 = rospy.Publisher('/hand2/reflex_takktile/command_motor_force', ForceCommand, queue_size=1)

    # Constantly capture the current hand state
    rospy.Subscriber('/hand2/reflex_takktile/hand_state', Hand, hand_state_cb)

    ##################################################################################################################    
    
    #'''
    print('Calibrate hand 1 ...')
    calibrate_fingers_hand1()
    print('Calibrate tactile 1 ... ')
    calibrate_tactile_hand1()
    #rospy.sleep(3)
    
    print('\nCalibrate hand 2') #Zero-ing the nagative value when loading
    calibrate_fingers_hand2()
    print('Calibrate tactile 2 ... ')
    calibrate_tactile_hand2()
    #'''
    rospy.sleep(1)
    
    
    #'''
    close_angle = 150
    preshape_angle = 60
    p0 = PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0))
    p1 = PoseCommand(deg2rad(close_angle), deg2rad(0), deg2rad(0), deg2rad(0))
    p2 = PoseCommand(deg2rad(0), deg2rad(close_angle), deg2rad(0), deg2rad(0))
    p3 = PoseCommand(deg2rad(0), deg2rad(0), deg2rad(close_angle), deg2rad(0))
    p4 = PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
    
    
    p_open1 = p0
    p_close1 = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(0))
    
    p_open2 =  PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
    p_close2 = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(preshape_angle))
    
    points = [p0, p1, p2, p3, p4, p_open1, p_close1, p_open1, p_close1, p_open1, p_close1, p_open2, p_close2, p_open2, p_close2, p_open2, p_close2, p0]
    #'''
    #points = [PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0))]
    
    hand1_thread = threading.Thread(
        target=move_thread,
        args=(pos_pub_hand1, "hand1", points)
    )
    
    hand2_thread = threading.Thread(
        target=move_thread,
        args=(pos_pub_hand2, "hand2", points)
    )

    #hand1_thread.daemon = True
    #hand2_thread.daemon = True
    
    #Start two thread in-order
    if concurrency == '0':
        hand1_thread.start()
        hand1_thread.join()
        hand2_thread.start()
        hand2_thread.join()
    else:    
        #Start two threads at the same time
        #'''
        hand1_thread.start()
        hand2_thread.start()
        hand1_thread.join()
        hand2_thread.join()
        #'''
    print('Done!')
    

def hand_state_cb(data):
    global hand_state
    hand_state = data


if __name__ == '__main__':
    main()
