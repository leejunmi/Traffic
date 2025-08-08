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


class traffic_info(IntEnum):
    # red = 0
    # green = 1
    # Invisible = 2
    # left_green = 3

    ##              best_train_custom2.pt
    green_arrow_and_green_arrow = 0
    red = 1
    green_and_green_arrow = 2
    green_arrow = 3
    green = 4
    yellow = 5
    red_and_yellow = 6
    green_and_yellow = 7 # 안쓰는게 
    red_and_green_arrow = 8


class Traffic():
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.result_image = None

        self.last_detected_class = None
        self.class_count = 0
        
        self.traffic_sign = 'Go'

        # PT_PATH = '/home/leejunmi/catkin_ws/src/vision/src/best_0729.pt'
        PT_PATH = '/home/leejunmi/catkin_ws/src/vision/src/best_train_custom2.pt' 

        self.model = YOLO(PT_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)
        # rospy.Subscriber('/self.current_s', float, self.s_callback, queue_size=5)

        self.current_s = 0.0
        self.stop_s = [5.0, 20.0, 30.0] # 3번 정지한다고 가정
        # 1.s값 구간 내에 있는지 확인
        # 2. 구간이면,,감지 시작 
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
        # 조건 다르게 해야하면 케이스 나누기 
        # if self.stop_s[0] - 5.0 < self.current_s < self.stop_s[0]:
        #     self.traffic_flag = True
        # elif self.stop_s[1] - 5.0 < self.current_s < self.stop_s[1]:
        #     self.traffic_flag = True
        # elif self.stop_s[2] - 5.0 < self.current_s < self.stop_s[2]:
        #     self.traffic_flag = True
        # else:
        #     self.traffic_flag = False

        self.traffic_flag = any(s - 10.0 < self.current_s < s for s in self.stop_s)
        print(f'traffic:{self.traffic_flag}')

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            # or
            # img = np.frombuffer(img.data, np.uint8)
            # self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # self.check_s()
            # if self.traffic_flag == True:
            self.yolo_detection(img)

        except Exception as e:
            rospy.logerr(f'callback error: {e}')

    def traffic_cnt_check(self, cls, conf, x1, y1, x2, y2):
        # cls: 클래스 number 

        if self.last_detected_class is None:
            self.last_detected_class = cls
            self.class_count = 1
            return
        
        if cls == self.last_detected_class:
            self.class_count += 1
        else:
            self.last_detected_class = cls
            self.class_count = 1
            return
            
        label = f'{traffic_info(cls).name} {conf:.2f}' 

        if self.class_count >= 2:
            print('count')
            '''0729'''
            # if cls == traffic_info.green: 
            #     self.traffic_pub.publish(Bool(data=False))
            #     print('Go')
            #     cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(self.result_image, label, (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # elif cls == traffic_info.red:
            #     self.traffic_pub.publish(Bool(data=True))
            #     print('Stop')
            #     cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #     cv2.putText(self.result_image, label, (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # elif cls == traffic_info.left_green:
            #     self.traffic_pub.publish(Bool(data=False))
            #     print('Left')
            #     cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            #     cv2.putText(self.result_image, label, (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            # elif cls == traffic_info.Invisible:
            #     # self.traffic_pub.publish(Bool(data=False))
            #     print('Invisible')
            #     cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            #     cv2.putText(self.result_image, label, (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            '''best_train'''
            if cls == traffic_info.green or cls == traffic_info.green_arrow_and_green_arrow: 
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

            elif cls == traffic_info.red_and_green_arrow:
                self.traffic_pub.publish(Bool(data=False))
                print('Red And Left_Green')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            elif cls == traffic_info.yellow:
                # self.traffic_pub.publish(Bool(data=False))
                print('yellow')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif cls == traffic_info.red_and_yellow:
                # self.traffic_pub.publish(Bool(data=False))
                print('red and yellow')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    

    def yolo_detection(self, img):
        self.result_image = img.copy()
        
        self.check_s()
        if self.traffic_flag == False:
                print('no traffic_s')
                cv2.imshow('YOLO Traffic Detection', self.result_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User pressed 'q'")
                return

        results = self.model(img)[0]

        ############ 컨피던스 가장 높은 1개만 판단
        # boxes = results.boxes
        # if len(boxes) == 0:
        #     cv2.imshow('YOLO Traffic Detection', self.result_image)
        #     return

        boxes = [b for b in results.boxes if (b.xyxy[0][2] - b.xyxy[0][0]) >= (b.xyxy[0][3] - b.xyxy[0][1])] # 세로가 더 긴 박스 제거
        if not boxes:
            print('조건 박스 없음')
            cv2.imshow('YOLO Traffic Detection', self.result_image)
            return
        ###########
        # boxes = sorted(results.boxes, key=lambda b: b.conf[0], reverse=True)
        # best_box = boxes[0]

        # x1, y1, x2, y2 = map(int, best_box.xyxy[0]) 
        # cls = int(best_box.cls[0])  # 클래스 인덱스
        # conf = float(best_box.conf[0])  # 신뢰도
        # label = f'{self.model.names[cls]} {conf:.2f}' # 출력용
        # # cls_name = self.model.names[cls]  
        # print(f'cls:{cls}, label:{label}')
        # self.traffic_cnt_check(cls, conf, x1,y1,x2,y2)

        ############# 전체 object 기준
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            cls = int(box.cls[0])  # 클래스 인덱스
            conf = float(box.conf[0])  # 신뢰도
            label = f'{self.model.names[cls]} {conf:.2f}'
            cls_name = self.model.names[cls] 
            self.traffic_cnt_check(cls, conf, x1,y1,x2,y2)

        cv2.imshow('YOLO Traffic Detection', self.result_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q'")



def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/output1.avi" 
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

####################
# 멀리 있는 신호등과 가까이 있는 신호등이 다른 경우
# -> 더 큰 신호등만 보도록 하기(이건 신호등뒤 Invisible로 될 수 있어서 위험)
# 같은 경우
# -> 컨피던스 기준으로 가장 큰 것만 보기

# 좌회전 신호 있는 경우(좌회전일때 Go 줘야하면) -> S값
# 아니면
# 좌회전 있는데 left 보내줘도 되면 -> S값 안 해도 됨. 좌회전일때 left 

# 신호등 깨지는 거 물어보기? 


######################################
# yellow 감지시 정지?
# 디텍 됨 + Invisible 일때? -> HSV로 판단(만약 좌회전이면 빨노 빨좌 .. )
# 주기 파악?
# 안 들어왔을 경우 예외처리? -> 횡단보도 ..?
# hsv? ->  red green만 보고 HSV 해도 되긴 함

'''신호등 인지 구간 or s값?'''
# 1. FOV 줄이고 멀리 본다면, 느리게 감속하는 거에 맞춰서 보자마자 감속 or 정지선의 s값 기반으로 인지구간 세워서 판단
# 2. FOV 적당하게 가까이 본다면, 바로 브레이크 or 




