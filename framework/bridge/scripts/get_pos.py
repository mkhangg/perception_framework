#!/usr/bin/env python  
import roslib
import rospy
import math
import tf

if __name__ == '__main__':
    rospy.init_node('rosie_tf_listener')
    listener = tf.TransformListener()


    rate = rospy.Rate(100.0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/base_link', '/left_hand', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        print('trans = ', type(trans), trans)
        print('rot = ', type(rot), rot)
       
        rate.sleep()