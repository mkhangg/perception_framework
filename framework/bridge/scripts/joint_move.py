#Enable robot before running test
# rosrun baxter_tools enable_robot.py -e

import rospy
import baxter_interface
import threading

def move_thread(limb, name, joint_pos):
    print(name, ' move to ', joint_pos)
    limb.move_to_joint_positions(joint_pos)

rospy.init_node('Baxter_Move')
left_limb = baxter_interface.Limb('left')
right_limb = baxter_interface.Limb('right')

left_angles = left_limb.joint_angles()
right_angles = right_limb.joint_angles()
print('Current left_angles = ', left_angles)
print('Current right_angles = ', right_angles)
#exit()

#left_limb.move_to_joint_positions(left_angles)
#right_limb.move_to_joint_positions(right_angles)

lpos0 = {'left_s0': 0, 'left_s1': 0, 'left_e0': 0, 'left_e1': 0, 'left_w0': 0, 'left_w1': 0, 'left_w2': 0}
rpos0 = {'right_s0':0, 'right_s1': 0, 'right_e0': 0, 'right_e1': 0, 'right_w0': 0, 'right_w1': 0, 'right_w2': 0}


left_thread = threading.Thread(
        target=move_thread,
        args=(left_limb, "left", lpos0)
    )
    
right_thread = threading.Thread(
    target=move_thread,
    args=(right_limb, "right", rpos0)
)

rospy.sleep(10)

left_thread.daemon = True
right_thread.daemon = True
left_thread.start()
right_thread.start()
left_thread.join()
right_thread.join()
print('Done!')
