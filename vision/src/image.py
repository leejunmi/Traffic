#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image


class ImagePub():
    def __init__(self):

        self.pub = rospy.Publisher('/image', Image, queue_size=5)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

    def time_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            print('No Image')
        
        frame = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        self.pub.publish(frame)
        rospy.loginfo('image publishing ... ')

def main(args=None):
    rospy.init_node('Publish_Image', anonymous=True) # init node 먼저 실행해야함, 공백 있으면 안 됨
    imagepub = ImagePub()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        imagepub.time_callback()
        rate.sleep()

    imagepub.cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()