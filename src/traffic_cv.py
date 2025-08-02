#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
바꿀것
cv: FOV 20
yolo: FOV 50, PT_PATH, VIDEO_MODE
'''
# left_green detect하는 버전
# 박스 기준으로 roi 해도 괜찮을듯

import rospy
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class TrafficLightDetector:
    ''' 원주 기준 HSV값 '''

    def __init__(self, image):
        cv2.imshow("image", image)
        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)

        height, width = image.shape[0], image.shape[1]

        self.roi = image[30:height-150,:]
    
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)

        self.red_lower = np.array([0, 120, 70])
        self.red_upper = np.array([10, 255, 255])

        self.red_lower2 = np.array([170, 200, 200])
        self.red_upper2 = np.array([180, 255, 255])

        self.yellow_lower = np.array([20, 100, 100])  
        self.yellow_upper = np.array([35, 255, 255])  

        self.green_lower = np.array([35, 115, 115]) 
        self.green_upper = np.array([100, 255, 255])  

        #####################
        self.red_mask = cv2.inRange(self.hsv, self.red_lower, self.red_upper)
        self.red_mask2 = cv2.inRange(self.hsv, self.red_lower2, self.red_upper2) # 쨍한 빨간색 추가
        self.red_mask = cv2.bitwise_or(self.red_mask, self.red_mask2)
        self.yellow_mask = cv2.inRange(self.hsv, self.yellow_lower, self.yellow_upper)
        self.green_mask = cv2.inRange(self.hsv, self.green_lower, self.green_upper)
        self.left_green_mask = np.zeros_like(self.green_mask)

        self.all_mask = cv2.bitwise_or(self.red_mask, self.green_mask)
        self.all_mask = cv2.bitwise_or(self.all_mask, self.yellow_mask)
        # self.all_mask = self.remove_small(self.all_mask, 20) # 작은 컴포넌트 제거하고 싶을때

        real_mask = cv2.bitwise_and(self.roi, self.roi, mask=self.all_mask)

        self.find_max(self.green_mask, real_mask)
        self.find_max_red(self.red_mask, real_mask)
        self.find_max_red(self.yellow_mask, real_mask)

        try:
            cv2.imshow("mask", real_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('no frame')
        except cv2.error as e:
            print(f'cv2 error: {e}')

    def find_max_red(self, img, real_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

        if num_labels > 1:
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 0번은 배경
            x, y, w, h, area = stats[max_label]
            cx, cy = centroids[max_label]

            if area > 130:
                # print(f"가장 큰 덩어리 중심: ({cx:.1f}, {cy:.1f}), 면적: {area}")
                # print(f"바운딩 박스: x={x}, y={y}, w={w}, h={h}")

                cv2.rectangle(real_mask, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.circle(real_mask, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    def find_max(self,img,real_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

        if num_labels > 1:
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 0번은 배경
            x, y, w, h, area = stats[max_label]
            cx, cy = centroids[max_label]

            if area > 130:
                print(f"가장 큰 덩어리 중심: ({cx:.1f}, {cy:.1f}), 면적: {area}")
                print(f"바운딩 박스: x={x}, y={y}, w={w}, h={h}")

                cv2.rectangle(real_mask, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.circle(real_mask, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                # 왼쪽 바운딩 박스
                diff = w+5
                cv2.rectangle(real_mask, (x-diff, y), (x + w - diff, y + h), (0, 255, 255), 2)

                x1 = max(x - diff, 0)
                x2 = min(x + w - diff, self.hsv.shape[1])
                y1 = max(y, 0)
                y2 = min(y + h, self.hsv.shape[0])
                self.left_green_roi = self.hsv[y1:y2, x1:x2]
                self.left_green_mask = cv2.inRange(self.left_green_roi, self.green_lower, self.green_upper)
                

    def remove_small(self, img: np.ndarray, min_num):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        for L in range(1, num_labels):
            if stats[L, cv2.CC_STAT_AREA] < min_num:
                img[labels == L] = 0
            elif stats[L, cv2.CC_STAT_AREA] > 1000:
                img[labels == L] = 0
            # print(stats[L, cv2.CC_STAT_AREA])
        return img
        
    def detect(self):
        print(f'red:{cv2.countNonZero(self.red_mask)}')
        print(f'green:{cv2.countNonZero(self.green_mask)}')
        print(f'yellow:{cv2.countNonZero(self.yellow_mask)}')

        mask_threshold = 180

        print()
        print('### Traffic ###')

        if cv2.countNonZero(self.red_mask) > mask_threshold:
            self.traffic_pub.publish(Bool(data=True))
            if cv2.countNonZero(self.green_mask) > mask_threshold:
                return 'red_and_green'
            elif cv2.countNonZero(self.yellow_mask) > mask_threshold:
                return 'red_and_yellow'
            else:
                return 'red'

        elif cv2.countNonZero(self.yellow_mask) > mask_threshold:
            self.traffic_pub.publish(Bool(data=False))
            return 'yellow'
         
        elif cv2.countNonZero(self.green_mask) > mask_threshold:
            self.traffic_pub.publish(Bool(data=False))
            if cv2.countNonZero(self.left_green_mask) > 50:
                return 'left_and_green'
            else:
                return 'green'
        
        else:
            self.traffic_pub.publish(Bool(data=False))
            return 'none'

class Traffic():
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.result_image = None

        self.last_detected_class = None
        self.class_count = 0
        
        self.traffic_sign = 'Go'

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
    

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")

            self.cv_detection(img)

        except Exception as e:
            rospy.logerr(f'callback error: {e}')

    def cv_detection(self,img):
        self.result_image = img.copy()

        detector = TrafficLightDetector(self.result_image)

        label_data = detector.detect()
        print(f'label: {label_data}')

        
def main():
    rospy.init_node('traffic_sign')

    # True = 영상 / False = morai
    VIDEO_MODE = True
    # VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/output2.avi" 
    VIDEO_PATH = "/home/leejunmi/VIDEO/output5.avi"
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
            traffic.cv_detection(frame)
            rate.sleep()
    else:
        rospy.spin()


if __name__ == '__main__':
    main()

