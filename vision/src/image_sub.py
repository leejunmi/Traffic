#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np

class ImageSub():
    def __init__(self):

        self.sub = rospy.Subscriber('/image', Image, self.image_callback)
        self.bridge = CvBridge()

        self.result_image = np.empty(shape=[1])

    def image_callback(self, data):
        self.result_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        cv2.imshow('img', self.result_image)
        cv2.waitKey(33)


def main(args=None):
    rospy.init_node('publish_image', anonymous=True) # init node 먼저 실행해야함, 공백 있으면 안 됨
    imagesub = ImageSub()
    try:
        rospy.spin() # callback 받기 기다림

    except:
        imagesub.cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()