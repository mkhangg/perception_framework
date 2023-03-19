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

def move_arm(name, move, str_reapeat):
    repeat = int(str_reapeat)
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
    print("============ Reference frame: %s" % arm.get_planning_frame())
    arm.set_end_effector_link(end_effector)
    print("============ Reference frame: %s" % arm.get_end_effector_link())

    robot = moveit_commander.RobotCommander()
    print("============ Robot Groups: ", robot.get_group_names())

  ## Sometimes for debugging it is useful to print the entire state of the
  ## robot.
  #print "============ Printing robot state"
  #print robot.get_current_state()
  #print "============"

   

    pos_pub_right = rospy.Publisher('/reflex_takktile/command_position', PoseCommand, queue_size=1)
    pos_pub_left = rospy.Publisher('/hand2/reflex_takktile/command_position', PoseCommand, queue_size=1)

    close_angle = 120
    preshape_angle = 0
    p_open =  PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
    p_close = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(preshape_angle))
  
  
    #current_pose = two_arms.get_current_pose(end_effector_link = end_effector)
    #print('current_pose = ', current_pose)
    #exit()
    
   
    # pose_target.orientation.w = 1
    delta = 0.4
    for i in range(repeat):
        print("Time = ", i)
        pose_target = geometry_msgs.msg.Pose()
        pose_target.position.x = 0.400
        pose_target.position.y = 0.700  #LEFT: y > 0
        pose_target.position.z = 0.70
        if name == "r":
            pose_target.position.y = -pose_target.position.y
      
        # pose_target.orientation.x = -0.124
        # pose_target.orientation.y = 0.991
        # pose_target.orientation.z = 0.031
        # pose_target.orientation.w = 0.028
        
        q = tf.transformations.quaternion_from_euler(deg2rad(0), deg2rad(90), deg2rad(0))   
        pose_target.orientation.x = q[0]
        pose_target.orientation.y = q[1]
        pose_target.orientation.z = q[2]
        pose_target.orientation.w = q[3]
        
        #Pos1: Pick
        if name == "l":
            pos_pub_left.publish(p_open) 
        else:
            pos_pub_right.publish(p_open)
        rospy.sleep(3)
        arm.set_pose_target(pose_target, end_effector_link = end_effector)
        print('pose_target 1 = ', pose_target)
        traj = arm.plan()
        #print("len traj 1 = ", len(traj.joint_trajectory.points))
        if move == "1":
            arm.execute(traj)
            if name == "l":
                pos_pub_left.publish(p_close)
            else:
                pos_pub_right.publish(p_close)
            rospy.sleep(3)

        #Pos2: Place
        
        pose_target2= copy.deepcopy(pose_target)
        pose_target2.position.x = pose_target.position.x + delta
        pose_target2.position.z = pose_target.position.z + (delta/1.0)
        arm.set_pose_target(pose_target2, end_effector_link = end_effector)
        print('pose_target 2 = ', pose_target2)
        #print("len traj 2 = ", len(traj.joint_trajectory.points))
        traj = arm.plan()
        if move == "1":
            arm.execute(traj)
            if name == "l":
                pos_pub_left.publish(p_open)
            else:
                pos_pub_right.publish(p_open)
        #rospy.sleep(3)

    moveit_commander.roscpp_shutdown()
    print "============ STOPPING"
    print('Exit....')
    exit()

import sys
if __name__=='__main__':
    try:
        move_arm(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
