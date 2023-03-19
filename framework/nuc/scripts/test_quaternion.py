import tf

import geometry_msgs
import math

def rad2deg(rad):
    return ((rad/math.pi)*180.0)

def deg2rad(deg):
    return ((deg/180.0)*math.pi)

#quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)

#quat = geometry_msgs.msg.Quaternion(0, 0, 0, 1) 
quat = [0, 0, -0.2, 0.98]
print(type(quat))
euler = tf.transformations.euler_from_quaternion(quat)
print(quat, " -> ", rad2deg(euler[0]), rad2deg(euler[1]), rad2deg(euler[2]))

quaternion = tf.transformations.quaternion_from_euler(0, 0, deg2rad(180))
print("quaternion = ", quaternion)