#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# FOV 40
# x:3.53, z:0.78, 나머지 0.00
# 깔끔한 레인만 됨

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class CvLanedetection(object):
    def __init__(self, video_mode=False, video_path=None):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher('/cv_lane', Float32, queue_size=1)

        self.steer = 0.0  # -11.0 ~ 11.0
        self.result_img = None
        self.real_img = None
        self.warp_img = None
        self.vis_img = None

        self.white_mask = None
        self.yellow_mask = None
        self.all_mask = None

        self.w = 640
        self.h = 480

        self.Minv = None
        self.pre_leftx = None
        self.pre_rightx = None

        self.current_leftX = None
        self.current_rightX = None

        self.mode = 0  # 1이면 left max steer, 2이면 right max steer
        self.r_count = 0
        self.l_count = 0

        self.unchanged_leftx = False
        self.unchanged_rightx = False

        self.video_mode = video_mode
        self.video_path = video_path

        if not self.video_mode:
            self.camera_img = rospy.Subscriber(
                '/image_jpeg3/compressed',
                CompressedImage,
                self.img_callback,
                queue_size=1
            )
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                rospy.logerr("Video open failed")
                raise RuntimeError("Cannot open video: {}".format(self.video_path))


    def img_callback(self, img_msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            self.process_frame(img)
        except Exception as e:
            rospy.logerr(f'Callback error: {e}')

    def histogram(self):
        """픽셀 합이 가장 큰 열(x)을 기준으로 슬라이딩 윈도우 탐색 시작점을 파악"""
        histogram = np.sum(self.warp_img[int(self.h/3):, :], axis=0)

        X_offset = 90                                                                  #### 수정
        right_offset = self.w//2 + X_offset

        left_slice = histogram[:self.w//2 - X_offset]
        right_slice = histogram[right_offset:]

        start_leftX = np.argmax(left_slice) if np.any(left_slice > 0) else -1
        start_rightX = (np.argmax(right_slice) + right_offset) if np.any(right_slice > 0) else 641

        self.find_lane(int(start_leftX), int(start_rightX))

    def find_lane(self, start_leftX, start_rightX):
        """start_x값 예외처리"""
        left_conditions = np.array([
            (self.pre_leftx is not None) and (abs(self.pre_leftx - start_leftX) >= 60) and (self.pre_leftx >= 0)
        ])
        left_values = np.array([self.pre_leftx if self.pre_leftx is not None else start_leftX])
        self.current_leftX = np.where(left_conditions, left_values, start_leftX)[0]

        right_conditions = np.array([
            (self.pre_rightx is not None) and (abs(self.pre_rightx - start_rightX) >= 60) and (self.pre_rightx <= 640)
        ])
        right_values = np.array([self.pre_rightx if self.pre_rightx is not None else start_rightX])
        self.current_rightX = np.where(right_conditions, right_values, start_rightX)[0]

        self.l_count = self.l_count + 1 if (self.pre_leftx == self.current_leftX and self.current_leftX != -1) else 0
        self.r_count = self.r_count + 1 if (self.pre_rightx == self.current_rightX and self.current_rightX != 641) else 0

        print(f'left:{self.current_leftX} right:{self.current_rightX}')

        if self.l_count > 1:
            self.unchanged_leftx = True
            self.l_count = 0
            self.current_leftX = start_leftX
        else:
            self.unchanged_leftx = False

        if self.r_count > 1:
            self.unchanged_rightx = True
            self.r_count = 0
            self.current_rightX = start_rightX
        else:
            self.unchanged_rightx = False

        self.pre_leftx = self.current_leftX
        self.pre_rightx = self.current_rightX

        self.steer = self.sliding_window(self.current_leftX, self.current_rightX)
        self.publish_steer()

    def publish_steer(self):
        assert isinstance(self.steer, float), 'lane steer type error(must float)'
        msg = Float32()
        msg.data = self.steer
        self.publisher.publish(msg)
        rospy.loginfo(f'steering count publishing: {msg.data}')
        print(' ')

    def sliding_window(self, current_leftX: int, current_rightX: int) -> float:
        left_firstX = current_leftX
        right_firstX = current_rightX
        num_windows = 15
        window_h = np.int16(self.warp_img.shape[0] / num_windows)
        nonzero = self.warp_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        min_num_pixel = 20
        window_margin = 55

        win_left_lane = []
        win_right_lane = []
        valid_left_win_count = 0
        valid_right_win_count = 0
        left_x = []
        right_x = []

        self.vis_img = cv2.cvtColor(self.warp_img, cv2.COLOR_GRAY2BGR)

        for window in range(num_windows):
            y_low = self.warp_img.shape[0] - (window + 1) * window_h
            y_high = self.warp_img.shape[0] - window * window_h
            lx_min = current_leftX - window_margin
            lx_max = current_leftX + window_margin
            rx_min = current_rightX - window_margin
            rx_max = current_rightX + window_margin

            cv2.rectangle(self.vis_img, (lx_min, y_low), (lx_max, y_high), (255, 255, 0), 2)
            cv2.rectangle(self.vis_img, (rx_min, y_low), (rx_max, y_high), (255, 255, 0), 2)

            left_inds = ((nonzeroy > y_low) & (nonzeroy < y_high) &
                         (nonzerox > lx_min) & (nonzerox < lx_max)).nonzero()[0]
            right_inds = ((nonzeroy > y_low) & (nonzeroy < y_high) &
                          (nonzerox > rx_min) & (nonzerox < rx_max)).nonzero()[0]
            win_left_lane.append(left_inds)
            win_right_lane.append(right_inds)

            if len(left_inds) > min_num_pixel:
                valid_left_win_count += 1
                current_leftX = int(np.percentile(nonzerox[left_inds], 90))
                # current_leftX = nonzerox[left_inds]

            if len(right_inds) > min_num_pixel:
                valid_right_win_count += 1
                current_rightX = int(np.percentile(nonzerox[right_inds], 10))
                # current_rightX = nonzerox[right_inds]

            left_x.append(current_leftX)
            right_x.append(current_rightX)

        # if 0 < valid_left_win_count < 3:
        #     print('왼쪽무시')
        #     current_leftX = -1
        #     win_left_lane = [np.array([], dtype=int) for _ in range(num_windows)]
        #     left_x = [-1] * 12
        #     cv2.putText(self.vis_img, '왼쪽 값 무시!!', (20, 105), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (200, 0, 0), 1, cv2.LINE_AA)
        # if 0 < valid_right_win_count < 3:
        #     print('오른쪽무시')
        #     current_rightX = 641
        #     win_right_lane = [np.array([], dtype=int) for _ in range(num_windows)]
        #     right_x = [641] * 12
        #     cv2.putText(self.vis_img, '오른쪽 값 무시!!', (20, 135), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (200, 0, 0), 1, cv2.LINE_AA)

        center_points = []
        for window in {1, 3, 7}:
            y_low = self.warp_img.shape[0] - (window + 1) * window_h
            y_high = self.warp_img.shape[0] - window * window_h

            lx = left_x[window]
            rx = right_x[window]

            cx = (lx + rx) // 2
            cy = (y_low + y_high) // 2

            pts = np.float32([[cx, cy]]).reshape(-1, 1, 2)
            real = cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2).astype(int)[0]
            cv2.circle(self.warp_img, (cx, cy), 2, (0, 0, 255), -1)
            cv2.circle(self.real_img, tuple(real), 3, (255, 0, 0), -1)
            center_points.append(tuple(real))

        cv2.line(self.vis_img, (self.w//2 - window_margin, 0), (self.w//2 - window_margin, self.h), (255,255,255), 1)
        cv2.line(self.vis_img, (self.w//2 + window_margin, 0), (self.w//2 + window_margin, self.h), (255,255,255), 1)

        result = self.calculate_encoder_count(center_points, current_leftX, left_firstX, current_rightX, right_firstX)
        return result

    def calculate_encoder_count(self, center_points:list, left_last_X:int, left_first_X:int, right_last_X:int, right_first_X:int):
        """center points(list)를 가지고 encoder count 계산"""
        conditions = {
            (right_last_X < 320 and abs(right_last_X-right_first_X) > 190) or (right_last_X < 320 and 320+150 > right_first_X and abs(right_last_X-right_first_X) > 160): 1,
            (320 < left_last_X and abs(left_first_X - left_last_X) > 190) or (320 < left_last_X and 320-150 < left_first_X and abs(left_first_X - left_last_X) > 160): 2
        }
        print(f'l diff:{abs(left_first_X - left_last_X)}')
        print(f'r diff:{abs(right_last_X-right_first_X)}')
        self.mode = conditions.get(True, 0)

        a = 13
        b = 7

        center_points = np.array(center_points)
        mean_center = np.mean(center_points[:,0])

        steer = (mean_center - self.w//2)/a

        first_center = center_points[2][0]
        last_center = center_points[0][0]
        diff = first_center - last_center

        if abs(diff) > 1.2 and steer * diff > 0:
            steer = steer + diff/b

        steer = np.clip(steer, -40.0, 40.0)
        rounded_steer = -round(steer, 1)

        real_mode_map = {1: 40.0, 2: -40.0}
        rounded_steer = real_mode_map.get(self.mode, -round(steer, 1))

        if self.mode == 1:
            rospy.loginfo(f'mode: {self.mode}, left steer')
            cv2.putText(self.real_img, f'self.mode:{self.mode}', (20,105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,0,0), 1, cv2.LINE_AA)
        elif self.mode == 2:
            rospy.loginfo(f'mode: {self.mode}, right steer')
            cv2.putText(self.real_img, f'self.mode:{self.mode}', (20,105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,0,0), 1, cv2.LINE_AA)

        self.mode = 0

        rounded_steer = rounded_steer*3

        if rounded_steer == 0:
            direction = 'straight'
        elif rounded_steer < 0:
            direction = 'right'
        else:
            direction = 'left'

        print(f'steer: {rounded_steer}, diff: {diff/b}, x:{round(((mean_center-320)/a),1)}')
        cv2.putText(self.real_img, f'x:{round(((mean_center-320)/16), 1)}, diff: {diff/a}', (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,20,120), 1, cv2.LINE_AA)
        cv2.putText(self.real_img, f'{direction}', (20,65), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (20,20,120), 2, cv2.LINE_AA)
        cv2.putText(self.real_img, f'{int(rounded_steer)}', (160,95), cv2.FONT_HERSHEY_TRIPLEX, 2.5,  (20,20,120), 2, cv2.LINE_AA)

        return rounded_steer


    def remove_sloped_and_small(self, canny_img: np.ndarray, min_num):
        h, w = canny_img.shape
        cleaned = canny_img.copy()
        cleaned[:h // 3 *2, :] = 0 # ROI 제한

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        for L in range(1, num_labels):
            ys, xs = np.where(labels == L)
            if len(xs) < 2:
                cleaned[labels == L] = 0
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            angle_rad = np.arctan2(vy, vx)
            angle_deg = np.degrees(angle_rad)
            near_horiz = abs(angle_deg) < 20 or abs(abs(angle_deg) - 180) < 20
            if near_horiz:
                cleaned[labels == L] = 0

        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        # for L in range(1, num_labels):
        #     if stats[L, cv2.CC_STAT_AREA] < min_num:
        #         cleaned[labels == L] = 0

        return cleaned

    def process_frame(self, img):
        self.real_img = img.copy()
        self.h, self.w, _ = self.real_img.shape

        # 터널이면 주석
        # self.real_img = self.img_process(self.real_img)

        g = cv2.cvtColor(self.real_img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        g = cv2.Canny(g, 70, 20)
        g = self.remove_sloped_and_small(g, 50)

        try:
            cv2.imshow('canny', g)
        except:
            pass

        h, w = g.shape[:2]

        src = np.float32([
            [200, 320],
            [w-200, 320],
            [10,  h-30],
            [w-10, h-30]
        ])
        for point in src:
            x, y = int(point[0]), int(point[1])
            cv2.circle(self.real_img, (x, y), 3, (0, 0, 255), -1)

        dst = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.warp_img = cv2.warpPerspective(g, M, (w, h))

        cv2.line(self.real_img, (0, self.real_img.shape[0]//2), (self.real_img.shape[1], self.real_img.shape[0]//2), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(self.real_img, (self.real_img.shape[1]//2, 0), (self.real_img.shape[1]//2, self.real_img.shape[0]), (255,255,255), 1, cv2.LINE_AA)

        self.histogram()

        try:
            cv2.imshow('result', self.real_img)
            cv2.imshow('vis', self.vis_img)
            cv2.waitKey(1)
        except:
            pass

    def img_process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.white_mask  = cv2.inRange(hsv,  np.array([0, 0, 200]),  np.array([190, 80, 255]))
        # 터널 안이면 np.array([0, 0, 100] 정도
        self.yellow_mask = cv2.inRange(hsv,  np.array([15,100,100]), np.array([30, 255,255]))
        self.all_mask = cv2.bitwise_or(self.white_mask, self.yellow_mask)

        return cv2.bitwise_and(img, img, mask=self.white_mask)


def main():
    rospy.init_node('cv_lane_node')
    VIDEO_MODE = True 
    VIDEO_PATH = "/home/leejunmi/VIDEO/lane3.avi"

    node = CvLanedetection(video_mode=VIDEO_MODE, video_path=VIDEO_PATH)

    if VIDEO_MODE:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = node.cap.read()
            if not ret:
                rospy.loginfo("영상 종료")
                break
            node.process_frame(frame)  
            rate.sleep()

        node.cap.release()
        cv2.destroyAllWindows()

    else:
        rospy.spin()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()