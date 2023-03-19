#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from reflex_msgs.msg import PoseCommand
import copy
import tf
import signal

b_running = True

def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False

import math
def deg2rad(deg):
    return ((deg*math.pi)/180.0)
  
from std_msgs.msg import String

def move_arm(move, str_reapeat):
    repeat = int(str_reapeat)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    two_arms = moveit_commander.MoveGroupCommander("both_arms")
    two_arms.set_pose_reference_frame(reference_frame="base_link")
    print("============ get_planning_frame : %s" % two_arms.get_planning_frame())
    #arm.set_end_effector_link(end_effector)
    print("============ get_end_effector_link: %s" % two_arms.get_end_effector_link())

    robot = moveit_commander.RobotCommander()
    print("============ Robot Groups: ", robot.get_group_names())

    pos_pub_right = rospy.Publisher('/reflex_takktile/command_position', PoseCommand, queue_size=1)
    pos_pub_left = rospy.Publisher('/hand2/reflex_takktile/command_position', PoseCommand, queue_size=1)

    close_angle = 120
    preshape_angle = 0
    p_open =  PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
    p_close = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(preshape_angle))

    delta = 0.4
    for i in range(repeat):
        global b_running
        if b_running == False:
            break
        print("Time = ", i)
        pose_targetL = geometry_msgs.msg.Pose()
        # pose_targetL.position.x = 0.579
        # pose_targetL.position.y = 0.4
        # pose_targetL.position.z = 0.885
        pose_targetL.position.x = 0.400
        pose_targetL.position.y = 0.700
        pose_targetL.position.z = 0.70
      
        q = tf.transformations.quaternion_from_euler(deg2rad(0), deg2rad(90), deg2rad(0))   
        pose_targetL.orientation.x = q[0]
        pose_targetL.orientation.y = q[1]
        pose_targetL.orientation.z = q[2]
        pose_targetL.orientation.w = q[3]
        
        pose_targetR = copy.deepcopy(pose_targetL)
        pose_targetR.position.y = -pose_targetL.position.y
        
        #pose_targetL = two_arms.get_current_pose(end_effector_link = "left_hand")
        print('pose_target L = ', pose_targetL)
        two_arms.set_pose_target(pose_targetL, end_effector_link = "left_hand")
        
        #pose_targetR = two_arms.get_current_pose(end_effector_link = "right_hand")
        print('pose_target R = ', pose_targetR)
        two_arms.set_pose_target(pose_targetR, end_effector_link = "right_hand")
        
        #Pos1: Pick
        traj = two_arms.plan()
        print("len traj 1 = ", len(traj.joint_trajectory.points))
        
        if move == "1":
            two_arms.execute(traj)
            pos_pub_left.publish(p_close)
            pos_pub_right.publish(p_close)
        rospy.sleep(3)
        #Pos2: Place
        pose_targetL2 = copy.deepcopy(pose_targetL)
        pose_targetL2.position.x = pose_targetL2.position.x + delta
        #pose_targetL2.position.y = pose_targetL2.position.y - (delta/1.0)
        pose_targetL2.position.z = pose_targetL2.position.z + delta
        
        pose_targetR2 = copy.deepcopy(pose_targetR)
        pose_targetR2.position.x = pose_targetR2.position.x + delta
        #pose_targetR2.position.y = pose_targetR2.position.y + (delta/1.0)
        pose_targetR2.position.z = pose_targetR2.position.z + delta
        
        print('pose_target2 L = ', pose_targetL2)
        two_arms.set_pose_target(pose_targetL2, end_effector_link = "left_hand")
        
        print('pose_target2 R = ', pose_targetR2)
        two_arms.set_pose_target(pose_targetR2, end_effector_link = "right_hand")
        
        traj = two_arms.plan()
        print("len traj 2 = ", len(traj.joint_trajectory.points))
        if move == "1":
            two_arms.execute(traj)
            pos_pub_left.publish(p_open)
            pos_pub_right.publish(p_open)
            #rospy.sleep(3)

    moveit_commander.roscpp_shutdown()
    print "============ STOPPING"
    print('Exit....')
    exit()

import sys
if __name__=='__main__':
    signal.signal(signal.SIGINT, handler)
    try:
        move_arm(sys.argv[1], sys.argv[2]) #Simulator, Repeat
    except Exception as e:
        print(e)
