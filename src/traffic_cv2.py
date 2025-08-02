#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class TrafficLightDetector:
    def __init__(self, image):
        cv2.imshow('original',image)
        self.traffic_pub = rospy.Publisher('/stop', Bool, queue_size=5)

        height, width = image.shape[:2]
        # 원주 default 30 ~ h-150
        self.roi = image[30:height - 150, :]
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)

        # HSV 범위
        self.red_lower = np.array([0, 120, 70])
        self.red_upper = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 200, 200])
        self.red_upper2 = np.array([180, 255, 255]) # 쨍한 빨간색 추가
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        self.green_lower = np.array([35, 115, 115])
        self.green_upper = np.array([100, 255, 255])

        self.red_mask1 = cv2.inRange(self.hsv, self.red_lower, self.red_upper)
        self.red_mask2 = cv2.inRange(self.hsv, self.red_lower2, self.red_upper2)
        self.red_mask = cv2.bitwise_or(self.red_mask1, self.red_mask2)
        self.yellow_mask = cv2.inRange(self.hsv, self.yellow_lower, self.yellow_upper)
        self.green_mask = cv2.inRange(self.hsv, self.green_lower, self.green_upper)
        self.left_green_mask = np.zeros_like(self.green_mask)

        self.final_decision = 'none'

        self.all_mask = cv2.bitwise_or(self.red_mask, self.yellow_mask)
        self.all_mask = cv2.bitwise_or(self.all_mask, self.green_mask)

        self.visual_img = cv2.bitwise_and(self.roi, self.roi, mask=self.all_mask)

        self.find_main_and_left_green()

        try:
            cv2.imshow("Traffic Detection", self.visual_img)
            cv2.waitKey(1)
        except:
            pass

    def find_main_and_left_green(self):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.all_mask)
        if num_labels <= 1: # 배경만있음
            self.final_decision = 'none'
            return

        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # 0이 배경
        x, y, w, h, area = stats[max_label]
        cx, cy = centroids[max_label]

        # print(f"중심: ({cx:.1f}, {cy:.1f}), 면적: {area}")
        # print(f"박스: x={x}, y={y}, w={w}, h={h}")

        cv2.rectangle(self.visual_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(self.visual_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        x1 = max(x, 0)
        x2 = min(x + w, self.hsv.shape[1])
        y1 = max(y, 0)
        y2 = min(y + h, self.hsv.shape[0])
        roi = self.hsv[y1:y2, x1:x2] 

        roi_red1 = cv2.inRange(roi, self.red_lower, self.red_upper)
        roi_red2 = cv2.inRange(roi, self.red_lower2, self.red_upper2)
        roi_red = cv2.bitwise_or(roi_red1, roi_red2)
        roi_yellow = cv2.inRange(roi, self.yellow_lower, self.yellow_upper)
        roi_green = cv2.inRange(roi, self.green_lower, self.green_upper)

        r = cv2.countNonZero(roi_red)
        y = cv2.countNonZero(roi_yellow)
        g = cv2.countNonZero(roi_green)

        print('#### ')
        print(f"red: {r}, yellow: {y}, green: {g}")

        thres = 50
        if r > thres and r > y and r > g:
            self.final_decision = 'red'
        elif y > thres and y > r and y > g:
            self.final_decision = 'yellow'
        elif g > thres and g > r and g > y:
            self.final_decision = 'green'
        else:
            self.final_decision = 'none'

        # left green ROI
        if self.final_decision == 'green' and area > thres:
            diff = w + 5
            lx1 = max(x - diff, 0)
            lx2 = min(x + w - diff, self.hsv.shape[1])
            ly1 = y1
            ly2 = y2

            cv2.rectangle(self.visual_img, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2) # green의 왼쪽 ROI

            left_roi = self.hsv[ly1:ly2, lx1:lx2]
            self.left_green_mask = cv2.inRange(left_roi, self.green_lower, self.green_upper)

    def detect(self):
        # print(f"### 최종 판단: {self.final_decision}")

        if self.final_decision == 'red':
            self.traffic_pub.publish(Bool(data=True)) # 원주 -> red는 일단 STOP
            if cv2.countNonZero(self.left_green_mask) > 30:
                return 'red_and_left_green'
            return 'red'

        elif self.final_decision == 'yellow':
            self.traffic_pub.publish(Bool(data=False))

            return 'yellow'

        elif self.final_decision == 'green':
            self.traffic_pub.publish(Bool(data=False))
            print(cv2.countNonZero(self.left_green_mask))
            if cv2.countNonZero(self.left_green_mask) > 30:
                return 'left_and_green'
        
            return 'green'

        else:
            self.traffic_pub.publish(Bool(data=False))
    
            return 'none'
        
class Traffic:
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.result_image = None

        self.video_mode = video_mode
        self.video_path = video_path

        if not self.video_mode:
            self.camera_img = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1)
        else:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                rospy.logerr("Video open failed")
                exit()

    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            self.cv_detection(img)
        except Exception as e:
            rospy.logerr(f'Callback error: {e}')

    def cv_detection(self, img):
        self.result_image = img.copy()
        detector = TrafficLightDetector(self.result_image)
        label = detector.detect()
        print(f"label: {label}")
        print(' ')


def main():
    rospy.init_node('traffic_sign')
    VIDEO_MODE = True # False: MORAI
    VIDEO_PATH = "/home/leejunmi/VIDEO/output1.avi"
    # VIDEO_PATH = '/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#5.avi'

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
