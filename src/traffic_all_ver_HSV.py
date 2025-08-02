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

from Traffic_Light_Detector import TrafficLightDetector


class traffic_info(IntEnum):
    # red = 0
    # green = 1
    # Invisible = 2
    # left_green = 3

    # best_train_custom2.pt
    # green_arrow_and_green_arrow = 0
    # red = 1
    # green_and_green_arrow = 2
    # green_arrow = 3
    # green = 4
    # yellow = 5
    # red_and_yellow = 6
    # green_and_yellow = 7
    # red_and_green_arrow = 8

    # HSV version
    red = 0
    green = 1
    yellow = 2
    red_and_green = 3
    red_and_yellow = 4
    all_green = 5
    left_green = 6
    none = 7

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

        self.video_mode = video_mode
        self.video_path = video_path

        if not self.video_mode: # morai
            self.camera_img = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1)
        else:
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                rospy.logerr("no video")
                exit()

    # def s_callback(self,msg):
    #     if self.current_s != 0.0:
    #         self.current_s = msg.data

    # def check_s(self):
    #     threshold = 10.0
    #     self.traffic_flag = any(s - threshold < self.current_s < s for s in self.stop_s)

    #     print(f'traffic:{self.traffic_flag}')
    
    def roi_img(self,img):
        height, width = img.shape[0], img.shape[1]

        # 상암
        img = img[0:height-200,:]

        # 원주 
        # img = img[0:height-200, width+30:width-30]
 
        return img

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            # or
            # img = np.frombuffer(img.data, np.uint8)
            # self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # self.check_s()
            # if self.traffic_flag == True:

            # roi_img = self.roi(img)
            self.yolo_detection(img)

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
                
        
    def yolo_detection(self, img):

        self.result_image = self.roi_img(img)
        
        # self.check_s()
        # if self.traffic_flag == False:
        # print('no traffic_s')
        
        cv2.imshow('YOLO Traffic Detection', self.result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q'")
            return

        results = self.model(img)[0]

        boxes = [b for b in results.boxes if (b.xyxy[0][2] - b.xyxy[0][0]) >= (b.xyxy[0][3] - b.xyxy[0][1])] # 세로가 더 긴 박스 제거
        if not boxes:
            print('조건 박스 없음')
            cv2.imshow('YOLO Traffic Detection', self.result_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User pressed 'q'")
            return
        
        # 컨피던스 기준
        # boxes = sorted(results.boxes, key=lambda b: b.conf[0], reverse=True)
        # best_box = boxes[0]

        # w-h*3 가 큰 거 기준
        boxes = sorted(
            boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) - 3 * (b.xyxy[0][3] - b.xyxy[0][1]),
            reverse=True)
        best_box = boxes[0]

        x1, y1, x2, y2 = map(int, best_box.xyxy[0]) 
        # cls = int(best_box.cls[0])  # 클래스 인덱스
        conf = float(best_box.conf[0])  # 신뢰도

        w = x2 - x1
        h = y2 - y1
        detector = TrafficLightDetector(self.result_image, x1, y1, w, h)

        label_data = detector.detect()
        print(f'label: {label_data}')

        # 카운트 사용
        self.traffic_cnt_check(traffic_info[label_data], conf, x1,y1,x2,y2)

        ############# 전체 object 기준
        # for box in results.boxes:
            # x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            # cls = int(box.cls[0])  # 클래스 인덱스
            # conf = float(box.conf[0])  # 신뢰도
            # label = f'{self.model.names[cls]} {conf:.2f}'
            # cls = self.model.names[cls] 
            # self.traffic_cnt_check(cls, label, x1,y1,x2,y2)

        cv2.imshow('YOLO Traffic Detection', self.result_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q'")


def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    VIDEO_PATH = "/home/leejunmi/VIDEO/output3.avi" 
    # VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/no_gps_NoObstacle#5.mp4"
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#5.avi'
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/스크린샷, 2025년 07월 28일 20시 35분 13초.webm'
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/스크린샷, 2025년 07월 28일 20시 33분 33초.webm'

    traffic = Traffic(video_mode=VIDEO_MODE, video_path=VIDEO_PATH)

    if VIDEO_MODE:
        rate = rospy.Rate(33)  
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

###################
# 원주
# FOV 30
# 상암
# 먼 신호등 볼 때 FOV 20?

####################
# 수정해야할것
# 1.중간에 디텍 안 될 때??
# 2.원주 기준 -> ROI 조절(너무 왼쪽. 오른쪽에 있는 박스 안 보도록)

# 어떤 신호등 있는지
# yellow 몇초인지
# 0nly_traffic_lane 써보기
# 
