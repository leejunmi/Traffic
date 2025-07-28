#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class Traffic():
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.result_image = None

        self.last_detected_class = None
        self.class_count = 0
        
        self.traffic_sign = 'Go'

        PT_PATH = '/home/leejunmi/Downloads/traffic_04_17.pt' 

        self.model = YOLO(PT_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)

        self.video_mode = video_mode
        self.video_path = video_path

        if not self.video_mode:
            self.camera_img = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1)
        else:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                rospy.logerr("비디오 열기 실패")
                exit()

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            # or
            # img = np.frombuffer(img.data, np.uint8)
            # self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            self.yolo_detection(img)

        except Exception as e:
            rospy.logerr(f'callback error: {e}')

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
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif cls == 'red':
                self.traffic_sign = '/Stop'
                self.traffic_pub.publish(Bool(data=True))   # stop = True
                print('Stop')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            
            self.class_count = 0
            self.last_detected_class = None

        

    def yolo_detection(self, img):
        self.result_image = img.copy()

        results = self.model(img)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            cls = int(box.cls[0])  # 클래스 인덱스
            conf = float(box.conf[0])  # 신뢰도
            label = f'{self.model.names[cls]} {conf:.2f}'
            cls = self.model.names[cls]

            self.traffic_cnt_check(cls, label, x1,y1,x2,y2)

        cv2.imshow('YOLO Traffic Detection', self.result_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q'")


def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    # VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#13.avi" 
    # VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/no_gps_NoObstacle#5.mp4"
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#5.avi'
    VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/스크린샷, 2025년 07월 28일 20시 35분 13초.webm'

    traffic = Traffic(video_mode=VIDEO_MODE, video_path=VIDEO_PATH)

    if VIDEO_MODE:
        rate = rospy.Rate(1)  
        while not rospy.is_shutdown():
            ret, frame = traffic.cap.read()
            if not ret:
                rospy.loginfo("영상 종료")
                break
            traffic.yolo_detection(frame)
            rate.sleep()
    else:
        rospy.spin()


if __name__ == '__main__':
    main()
