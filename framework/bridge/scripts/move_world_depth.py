#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
#import geometry_msgs.msg.Quaternion
from reflex_msgs.msg import PoseCommand
import tf
import copy

import math
def deg2rad(deg):
    return ((deg*math.pi)/180.0)
  
from std_msgs.msg import String

#Example: move_arm(r, 1, 1)
def move_arm(name):
    if name == "l":
        arm_name = "left_arm"
        end_effector = "left_hand"
    else:
        arm_name = "right_arm"
        end_effector = "right_hand"

    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    arm = moveit_commander.MoveGroupCommander(arm_name)
    #arm.set_planning_frame('base_link')	
    print("============ Reference frame: %s" % arm.get_planning_frame())
    arm.set_end_effector_link(end_effector)
    print("============ End effector frame: %s" % arm.get_end_effector_link())

    robot = moveit_commander.RobotCommander()
    print("============ Robot Groups: ", robot.get_group_names())
    
    pose_target = geometry_msgs.msg.Pose()
    base = 0.35
    pose_target.position.x = base
    pose_target.position.y = 0.0  #LEFT: y > 0
    pose_target.position.z = 1.2
    
    # pose_target.position.x = 0.7
    # pose_target.position.y = 0.3  #LEFT: y > 0
    # pose_target.position.z = 1.3
    # if name == "r":
        # pose_target.position.y = -pose_target.position.y
    
    q = tf.transformations.quaternion_from_euler(deg2rad(0), deg2rad(0), deg2rad(0))   
    pose_target.orientation.x = q[0]
    pose_target.orientation.y = q[1]
    pose_target.orientation.z = q[2]
    pose_target.orientation.w = q[3]
    
    n = 6
    delta = 0.05
    print('Forward')
    for i in range(n):
        pose_target.position.x = base + delta*i
        arm.set_pose_target(pose_target, end_effector_link = end_effector)
        print(pose_target)
        traj = arm.plan()
        #print("len traj 1 = ", len(traj.joint_trajectory.points))
        arm.execute(traj)
        rospy.sleep(3)
    
    dist = base + delta*(n-1)
    print('Backward')
    for i in range(n):
        pose_target.position.x = dist - delta*i
        arm.set_pose_target(pose_target, end_effector_link = end_effector)
        print(pose_target)
        traj = arm.plan()
        #print("len traj 1 = ", len(traj.joint_trajectory.points))
        arm.execute(traj)
        rospy.sleep(3)

    moveit_commander.roscpp_shutdown()
    print "============ STOPPING"
    print('Exit....')
    exit()

import sys
if __name__=='__main__':
    try:
        move_arm(sys.argv[1]) # r
    except Exception as e:
        print(e)
