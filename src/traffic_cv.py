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
from enum import IntEnum

# from Traffic_Light_Detector import TrafficLightDetector

#from ultralytics import YOLO


class TrafficLightDetector:
    def __init__(self, image, x, y, w, h):

        height, width = image.shape[0], image.shape[1]

        self.roi = image[0:height-250,:]
        # self.roi = image[y+h_roi:y+h-h_roi, x:x+w]
        # self.roi1 = image[y+h_roi:y+h-h_roi, x+w//12:x+w*3//12]
        # self.roi2 = image[y+h_roi:y+h-h_roi, x+w*4//12:x+w*6//12]
        # self.roi3 = image[y+h_roi:y+h-h_roi, x+w*6//12:x+w*8//12]
        # self.roi4 = image[y+h_roi:y+h-h_roi, x+w*9//12:x+w]

        try:
            cv2.imshow('ROI', self.roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('no frame')
        except cv2.error as e:
            print(f'cv2 error: {e}')
            
    
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)

        self.red_lower = np.array([0, 120, 70])
        self.red_upper = np.array([10, 255, 255])

        self.red_lower2 = np.array([170, 200, 200])
        self.red_upper2 = np.array([180, 255, 255])

        self.yellow_lower = np.array([20, 70, 70])  
        self.yellow_upper = np.array([35, 255, 255])  

        self.green_lower = np.array([35, 100, 100]) 
        self.green_upper = np.array([100, 255, 255])  

        self.red_mask = cv2.inRange(self.hsv, self.red_lower, self.red_upper)
        # self.red_mask2 = cv2.inRange(self.hsv, self.red_lower2, self.red_upper2) # 쨍한 빨간색 추가
        # self.red_mask = cv2.bitwise_or(self.red_mask, self.red_mask2)

        self.yellow_mask = cv2.inRange(self.hsv, self.yellow_lower, self.yellow_upper)
        self.green_mask = cv2.inRange(self.hsv, self.green_lower, self.green_upper)

        # self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_HSV2BGR)

        self.all_mask = cv2.bitwise_or(self.red_mask, self.green_mask)

        cv2.imshow("mask", cv2.bitwise_and(self.roi, self.roi, mask=self.all_mask))

        # cv2.imshow("red",self.red_mask)

        
    def detect(self):
        print(f'red:{cv2.countNonZero(self.red_mask)}')
        print(f'green:{cv2.countNonZero(self.green_mask)}')

        if cv2.countNonZero(self.red_mask) > 2:
            print('****')
            if cv2.countNonZero(self.green_mask) > 0:
                return 'red_and_green'
            elif cv2.countNonZero(self.yellow_mask) > 0:
                return 'red_and_yellow'
            else:
                return 'red'

        elif cv2.countNonZero(self.yellow_mask) > 0:
            return 'yellow'
         
        # elif cv2.countNonZero(self.green_mask) > 5:
        #     if cv2.countNonZero(self.green_mask) > 3:
        #         return 'all_green'
        #     else:
        #         return 'green'
        # elif cv2.countNonZero(self.green_mask_sign) > 0:
        #     return 'left_green'
        
        else:
            return 'none'


class Traffic():
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.result_image = None

        self.last_detected_class = None
        self.class_count = 0
        
        self.traffic_sign = 'Go'

        # PT_PATH = '/home/macaron/best_train_custom2.pt'
        # PT_PATH = '/home/macaron/traffic_04_17.pt'
        PT_PATH = '/home/leejunmi/catkin_ws/src/vision/src/best_train_custom2.pt'

        self.model = YOLO(PT_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)
        # rospy.Subscriber('/self.current_s', float, self.s_callback, queue_size=5)

        self.current_s = 0.0
        self.stop_s = [5.0, 20.0, 30.0] # 3번 정지한다고 가정

        self.traffic_flag = False

        self.video_mode = video_mode
        self.video_path = video_path

        if not self.video_mode: # morai
            self.camera_img = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1)
        else:
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                rospy.logerr("no video")
                exit()

    def s_callback(self,msg):
        if self.current_s != 0.0:
            self.current_s = msg.data

    def check_s(self):
        self.traffic_flag = any(s - 5.0 < self.current_s < s for s in self.stop_s)
        print(f'traffic:{self.traffic_flag}')
    
    def roi(img):
        height, width = img.shape[0], img.shape[1]
        img = img[0:height-150,:]


    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")

            # roi_img = self.roi(img)
            # self.yolo_detection(roi_img)
            self.cv_detection(img)

        except Exception as e:
            rospy.logerr(f'callback error: {e}')

    def traffic_cnt_check(self, cls, conf, x1, y1, x2, y2):
        # HSV version
        if self.last_detected_class is None:
            self.last_detected_class = cls
            self.class_count = 1
            return
        
        if cls == self.last_detected_class:
            self.class_count += 1
        else:
            if self.class_count < 2:
                return
            else: # 이전 클래스 2회 이상이면 업데이트
                self.last_detected_class = cls
                self.class_count = 1
                return 
            
        label = f'{traffic_info(cls).name} {conf:.2f}' 

        if self.class_count >= 2:
            '''0729'''
            if cls == traffic_info.green: 
                self.traffic_pub.publish(Bool(data=False))
                print('Go')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif cls == traffic_info.red:
                self.traffic_pub.publish(Bool(data=True))
                print('Stop')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            elif cls == traffic_info.yellow:
                self.traffic_pub.publish(Bool(data=True))
                print('ready to Stop')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif cls == traffic_info.none:
                self.traffic_pub.publish(Bool(data=False))
                print('Invisible')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #################
            
            elif cls == traffic_info.red_and_green:
                self.traffic_pub.publish(Bool(data=False))
                print('red_and_green')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            elif cls == traffic_info.red_and_yellow:
                self.traffic_pub.publish(Bool(data=False))
                print('red_and_yellow')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif cls == traffic_info.all_green:
                self.traffic_pub.publish(Bool(data=True))
                print('all_green')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    def cv_detection(self,img):
        self.result_image = img.copy()

        detector = TrafficLightDetector(self.result_image, 1,1,1,1)

        label_data = detector.detect()
        print(f'label: {label_data}')

        

def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/output2.avi" 
    # VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/no_gps_NoObstacle#5.mp4"
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#5.avi'
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/스크린샷, 2025년 07월 28일 20시 35분 13초.webm'
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/스크린샷, 2025년 07월 28일 20시 33분 33초.webm'

    traffic = Traffic(video_mode=VIDEO_MODE, video_path=VIDEO_PATH)

    if VIDEO_MODE:
        rate = rospy.Rate(1)  
        while not rospy.is_shutdown():
            ret, frame = traffic.cap.read()
            if not ret:
                rospy.loginfo("영상 종료")
                break
            traffic.cv_detection(frame)
            rate.sleep()
    else:
        rospy.spin()


if __name__ == '__main__':
    main()

