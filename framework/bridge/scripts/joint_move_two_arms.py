#Enable robot before running test
# rosrun baxter_tools enable_robot.py -e

import rospy
import baxter_interface
import threading

def move_thread(limb, joint_pos1, joint_pos2):
    for _move in range(1):
        limb.move_to_joint_positions(joint_pos1)
        limb.move_to_joint_positions(joint_pos2)

rospy.init_node('Hello_Baxter')
left_limb = baxter_interface.Limb('left')
right_limb = baxter_interface.Limb('right')

left_angles = left_limb.joint_angles()
right_angles = right_limb.joint_angles()
print('left_angles = ', left_angles)
print('right_angles = ', right_angles)

left_limb.move_to_joint_positions(left_angles)
right_limb.move_to_joint_positions(right_angles)
#exit()
rpos1 = {'right_s0': -0.459, 'right_s1': -0.202, 'right_e0': 1.807, 'right_e1': 1.714, 'right_w0': -0.906, 'right_w1': -1.545, 'right_w2': -0.276}
rpos2 = {'right_s0': -0.395, 'right_s1': -0.202, 'right_e0': 1.831, 'right_e1': 1.981, 'right_w0': -1.979, 'right_w1': -1.100, 'right_w2': -0.448}

lpos1 = {'left_s0': -0.459, 'left_s1': -0.202, 'left_e0': 1.807, 'left_e1': 1.714, 'left_w0': -0.906, 'left_w1': -1.545, 'left_w2': -0.276}
lpos2 = {'left_s0': -0.395, 'left_s1': -0.202, 'left_e0': 1.831, 'left_e1': 1.981, 'left_w0': -1.979, 'left_w1': -1.100, 'left_w2': -0.448}


left_thread = threading.Thread(
        target=move_thread,
        args=(left_limb, lpos1, lpos2)
    )
    
right_thread = threading.Thread(
    target=move_thread,
    args=(right_limb, rpos1, rpos2)
)

#left_thread.daemon = True
#right_thread.daemon = True
left_thread.start()
right_thread.start()
left_thread.join()
right_thread.join()
print('Done!')
