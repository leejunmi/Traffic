#!/usr/bin/env python3
# -*-coding:utf-8-*-
''' 좌회전 있으면 -> s값 혹은 left 보내주기
stop일때만 stop 보내기 '''

import rospy
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

class Traffic():
    def __init__(self):
        self.bridge = CvBridge()

        self.camera_img = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1)
        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)
        self.result_image = None

        PT_PATH = '/home/leejunmi/Downloads/traffic_04_17.pt'

        self.model = YOLO(PT_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # self.green_cnt = 0
        # self.red_cnt = 0

        self.last_detected_class = None
        self.class_count = 0
        self.traffic_sign = 'Go'

    def img_callback(self, img):

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")

            # or
            # img = np.frombuffer(img.data, np.uint8)
            # self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            self.yolo_detection(img)

        except Exception as e:
            print(f'error: {e}')
    
    # def traffic_cnt_check(self, cls,  x1,y1,x2,y2):
    #     if cls == 'green':
    #         self.green_cnt += 1
    #         self.red_cnt = 0
    #     elif cls == 'red':
    #         self.red_cnt += 1
    #         self.green_cnt = 0


    #     if self.green_cnt > 1:
    #         self.traffic_sign = '/Go'
    #         self.green_cnt = 0
    #         self.red_cnt = 0
    #         cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(self.result_image, cls, (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #         self.traffic_pub.publish(Bool(data=False)) 
    #         print('Go')

    #     elif self.red_cnt > 1:
    #         self.green_cnt = 0
    #         self.red_cnt = 0
    #         cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(self.result_image, cls, (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #         self.traffic_pub.publish(Bool(data=True)) 
    #         print('stop')

    def traffic_cnt_check(self, cls, label, x1, y1, x2, y2):
        if cls == self.last_detected_class:
            self.class_count += 1
        else:
            self.class_count = 1
            self.last_detected_class = cls

        if self.class_count > 1:
            if cls == 'green':
                self.traffic_pub.publish(Bool(data=False))  # stop = False
                print('Go')

            elif cls == 'red':
                self.traffic_sign = '/Stop'
                self.traffic_pub.publish(Bool(data=True))   # stop = True
                print('Stop')

            cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.result_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.class_count = 0
            self.last_detected_class = None



    def yolo_detection(self,img):
        self.result_image = img.copy()

        results = self.model(img)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            cls = int(box.cls[0])  # 클래스 인덱스
            conf = float(box.conf[0])  # 신뢰도
            label = f'{self.model.names[cls]} {conf:.2f}'
            cls = self.model.names[cls]

            self.traffic_cnt_check(cls, label, x1,y1,x2,y2)


            # cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(self.result_image, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('YOLO_DETECTION', self.result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('no frame')


def main():
    rospy.init_node('traffic_sign')

    traffic = Traffic()
    # while not rospy.is_shutdown():
    #    traffic.yolo_detection()

    rospy.spin()


if __name__ == '__main__':
    main()

        








    
