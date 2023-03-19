#!/usr/bin/env python
# license removed for brevity

import rospy

# Brings in the SimpleActionClient
import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from  nav_msgs.msg import Odometry

current_pose = None
def callback_odom(odom_data): #250Hz
    global current_pose
    current_pose = odom_data.pose.pose
    #print(current_pose)

import math
import tf
def deg2rad(deg):
    return ((deg/180.0)*math.pi)

def move_base(x, y, rot):
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

   # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    #Copy orientation and z from current_pose

    pose = tf.transformations.quaternion_from_euler(0, 0, deg2rad(rot))
    goal.target_pose.pose.orientation.x = pose[0]
    goal.target_pose.pose.orientation.y = pose[1]
    goal.target_pose.pose.orientation.z = pose[2]
    goal.target_pose.pose.orientation.w = pose[3]

   # Sends the goal to the action server.
    client.send_goal(goal)
    wait = client.wait_for_result()
   # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
    # Result of executing the action
        return client.get_result()   

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
        rospy.init_node('movebase_client_py')
        #rospy.Subscriber(f'/odom', Odometry, callback_odom)
        #             X
        #             |   
        #             |
        # Y(+)------------------ Y(-)
        x_mid, y_mid = 4.0, -0.5
        rospy.sleep(5)
        move_i = 1
        print(f"Move {move_i}: ", move_base(3, y_mid, 0))
        move_i += 1
        rospy.sleep(1)
        print("Move {move_i}: ", move_base(7, y_mid, 0))
        move_i += 1
        rospy.sleep(1)
        print("Move {move_i}: ", move_base(7, y_mid-2, -90))
        move_i += 1
        rospy.sleep(3)
        print("Move {move_i}: ", move_base(7, y_mid, 180))
        move_i += 1
        rospy.sleep(1)
        print("Move {move_i}: ", move_base(3, y_mid, 180))
        move_i += 1
        rospy.sleep(1)
        print("Move {move_i}: ", move_base(0, 0, 0))
        #print(move_base(5.5, y_mid))
        #print(move_base(6, y_mid-2))
        #result = movebase_client()
        # if result:
        #     rospy.loginfo("Goal execution done!")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
