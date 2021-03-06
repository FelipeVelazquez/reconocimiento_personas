#!/usr/bin/env python
# license removed for brevity
import rospy
from reconocimiento_personas.msg import coordenadas

def talker():
    pub = rospy.Publisher('chatter', coordenadas)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(50) # 10hz
    msg = coordenadas()
    msg.y = 50
    msg.h = 400
    msg.x = 100
    msg.w = 400
    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
