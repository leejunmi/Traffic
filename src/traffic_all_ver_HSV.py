#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 최종 수정
'''
바꿀것
cv: FOV 20
yolo: FOV 50, PT_PATH, VIDEO_MODE
'''

import rospy
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from enum import IntEnum



class TrafficLightDetector:
    def __init__(self, image, x, y, w, h):
        h_roi  = h//3
        # print(f'w,h:{w},{h}')

        # if h*3 > w: # 3구
        
        self.roi = image[y+h_roi:y+h-h_roi, x:x+w]
        self.roi1 = image[y+h_roi:y+h-h_roi, x+w//12:x+w*3//12]
        self.roi2 = image[y+h_roi:y+h-h_roi, x+w*4//12:x+w*6//12]
        self.roi3 = image[y+h_roi:y+h-h_roi, x+w*6//12:x+w*8//12]
        self.roi4 = image[y+h_roi:y+h-h_roi, x+w*9//12:x+w]

        try:
            cv2.imshow('ROI', self.roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('no frame')
        except cv2.error as e:
            print(f'cv2 error: {e}')
            
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.hsv1 = cv2.cvtColor(self.roi1, cv2.COLOR_BGR2HSV)
        self.hsv2 = cv2.cvtColor(self.roi2, cv2.COLOR_BGR2HSV)
        self.hsv3 = cv2.cvtColor(self.roi3, cv2.COLOR_BGR2HSV)
        self.hsv4 = cv2.cvtColor(self.roi4, cv2.COLOR_BGR2HSV)

        self.red_lower = np.array([0, 120, 70])
        self.red_upper = np.array([10, 255, 255])

        self.red_lower2 = np.array([170, 200, 200])
        self.red_upper2 = np.array([180, 255, 255])

        self.yellow_lower = np.array([20, 70, 70])  
        self.yellow_upper = np.array([35, 255, 255])  

        self.green_lower = np.array([35, 80, 80]) 
        self.green_upper = np.array([100, 255, 255])  

        self.red_mask = cv2.inRange(self.hsv1, self.red_lower, self.red_upper)
        self.red_mask2 = cv2.inRange(self.hsv1, self.red_lower2, self.red_upper2) # 쨍한 빨간색 추가

        self.red_mask = cv2.bitwise_or(self.red_mask, self.red_mask2)

        self.yellow_mask = cv2.inRange(self.hsv2, self.yellow_lower, self.yellow_upper)

        self.green_mask_sign = cv2.inRange(self.hsv3, self.green_lower, self.green_upper)
        self.green_mask = cv2.inRange(self.hsv4, self.green_lower, self.green_upper)
        # print(self.yellow_mask)
        
    def detect(self):

        if cv2.countNonZero(self.red_mask) > 2:
            print('****')
            if cv2.countNonZero(self.green_mask_sign) > 0:
                return 'red_and_green'
            elif cv2.countNonZero(self.yellow_mask) > 0:
                return 'red_and_yellow'
            else:
                return 'red'

        elif cv2.countNonZero(self.yellow_mask) > 0:
            return 'yellow'
         
        elif cv2.countNonZero(self.green_mask) > 0:
            if cv2.countNonZero(self.green_mask_sign) > 0:
                return 'all_green'
            else:
                return 'green'
        elif cv2.countNonZero(self.green_mask_sign) > 0:
            return 'left_green'
        
        else:
            return 'none'
''''''
class traffic_info(IntEnum):
    # red = 0
    # green = 1
    # Invisible = 2
    # left_green = 3

    ##              best_train_custom2.pt
    # green_arrow_and_green_arrow = 0
    # red = 1
    # green_and_green_arrow = 2
    # green_arrow = 3
    # green = 4
    # yellow = 5
    # red_and_yellow = 6
    # green_and_yellow = 7 # 안쓰는게 
    # red_and_green_arrow = 8

    ## HSV version
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

        # PT_PATH = '/home/macaron/best_train_custom2.pt'
        # PT_PATH = '/home/macaron/best_0729.pt'
        PT_PATH = '/home/leejunmi/catkin_ws/src/vision/src/best_train_custom2.pt'

        self.model = YOLO(PT_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)

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

    
    def roi_img(self,img):
        height, width = img.shape[0], img.shape[1]

        # 상암
        # img = img[0:height-200,:]

        # 원주 
        img = img[0:height-150, 30:width-30]
 
        return img

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")

            self.yolo_detection(img)

        except Exception as e:
            rospy.logerr(f'callback error: {e}')    

    def traffic_cnt_check(self, cls, x1, y1, x2, y2):

        print(f"[DEBUG] 현재 cls: {cls} ({traffic_info(cls).name}), "
          f"이전 cls: {self.last_detected_class} "
          f"({traffic_info(self.last_detected_class).name if self.last_detected_class is not None else 'None'}), "
          f"count: {self.class_count}")
        

        # HSV version
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
            
        label = f'{traffic_info(cls).name} ' 

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
                self.traffic_pub.publish(Bool(data=False))
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
            elif cls == traffic_info.all_green:
                self.traffic_pub.publish(Bool(data=False))
                print('all_green')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            #################
            
            elif cls == traffic_info.red_and_green:
                self.traffic_pub.publish(Bool(data=False))
                print('red_and_green')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            elif cls == traffic_info.red_and_yellow:
                self.traffic_pub.publish(Bool(data=True))
                print('red_and_yellow')
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(self.result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
                
        
    def yolo_detection(self, img):
        # cv2.imshow('origin', img)
        # print(f'img shape: {img.shape}')

        self.result_image = self.roi_img(img)
        # self.result_image = img.copy()
        

        results = self.model(self.result_image)[0]

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
        conf = float(best_box.conf[0])  # 신뢰도

        w = x2 - x1
        h = y2 - y1
        detector = TrafficLightDetector(self.result_image, x1, y1, w, h)

        label_data = detector.detect()
        print(f'label: {label_data}')

        # 카운트 사용
        self.traffic_cnt_check(traffic_info[label_data], x1,y1,x2,y2)


        # ############# 전체 object 기준
        # for box in results.boxes:
        #     x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        #     cls = int(box.cls[0])  # 클래스 인덱스
        #     conf = float(box.conf[0])  # 신뢰도
        #     label = f'{self.model.names[cls]} {conf:.2f}'
        #     cls = self.model.names[cls] 
        #     # self.traffic_cnt_check(cls, label, x1,y1,x2,y2)

        #     cv2.rectangle(self.result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(self.result_image, label, (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('YOLO Traffic Detection', self.result_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q'")


def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    VIDEO_PATH = "/home/leejunmi/VIDEO/output1.avi" 
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

#