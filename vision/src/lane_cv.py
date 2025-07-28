#!/usr/bin/env python3
# 0711

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float32
from math import *

class CvLanedetection():
    def __init__(self):
        
        self.publisher = rospy.Publisher('/cv_lane', Float32, queue_size=5)


        self.steer = 0.0 # 최종 publihs할 encoder count 값, -11.0~11.0
 
        self.result_img = None 
        self.real_img = None # 결과 이미지(원본이미지에 덮어씌울 이미지)
        self.warp_img = None # window 표시용 warp 이미지
        self.vis_img = None

        self.w = 640 
        self.h = 360

        self.Minv = None
        self.pre_leftx = None
        self.pre_rightx = None # 초기값 넣지말기

        self.current_leftX = None
        self.current_rightX = None

        self.mode = 0 # 1이면 왼쪽으로 최대steer(11), 2이면 오른쪽으로 최대 스티어(-11)
        self.r_count = 0
        self.l_count = 0

        self.unchanged_leftx = False
        self.unchanged_rightx = False

        # web cam
        # self.video_path = '/home/leejunmi/ros2_ws/src/lane_detection/scripts/0629.mov'
        # self.video_path = '/home/leejunmi/ros2_ws/src/lane_detection/scripts/0624 (1).mov'
        # self.video_path = '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_D6CAAAAF-video-index0' # 카메라 usb 번호
        # self.video_path = "/home/leejunmi/output_GPS13#_NoObstacle-case1.mp4"
        self.video_path = "/home/leejunmi/Downloads/output_video_lane_detected.avi"
        

    def histogram(self):
        '''픽셀 합이 가장 큰 열(x)을 기준으로 슬라이딩 윈도우 탐색 시작점을 파악'''

        histogram = np.sum(self.warp_img[int(self.h/3):, :], axis=0) 
        
		# 파라미터
        X_offset = 90 # 중앙에 노이즈 들어오고 커브 안 빡세면 키우기
        right_offset = self.w//2 + X_offset

        start_leftX = np.argmax(histogram[:self.w//2 - X_offset]) if np.any(histogram[:self.w//2 - X_offset]) > 0 else -1 # 히스토그램 왼쪽 최대 구하기(시작점), 픽셀 없으면 -1
        start_rightX = np.argmax(histogram[right_offset:]) + right_offset if np.any(histogram[right_offset:] > 0) else 641 # 히스토그램 오른쪽 최대 구하기(시작점), 픽셀 없으면 641

        self.find_lane(int(start_leftX), int(start_rightX)) # <class 'numpy.int64'> -> Int


    def find_lane(self, start_leftX, start_rightX):
        '''start_x값 예외처리'''

        # 왼쪽 Lane 예외 처리
        left_conditions = np.array([self.pre_leftx and (abs(self.pre_leftx - start_leftX) >= 60) and (self.pre_leftx >= 0)]) # 값 튀는 거 방지
        left_values = np.array([self.pre_leftx])
        self.current_leftX = np.where(left_conditions, left_values, start_leftX)[0]

        # 오른쪽 Lane 예외 처리
        right_conditions = np.array([self.pre_rightx and ((abs(self.pre_rightx - start_rightX) >= 60)) and (self.pre_rightx <= 640)])
        right_values = np.array([self.pre_rightx])
        self.current_rightX = np.where(right_conditions, right_values, start_rightX)[0]

        # 특정 값으로 유지될 때 start_x 값으로 보도록 재탐색
        self.l_count = self.l_count + 1 if (self.pre_leftx==self.current_leftX and self.current_leftX != -1) else 0
        self.r_count = self.r_count + 1 if (self.pre_rightx==self.current_rightX and self.current_rightX != 641) else 0

        print(f'left:{self.current_leftX} right:{self.current_rightX}')

        # 값이 변하지 않을 때 재탐색
        if self.l_count > 1:
            # print(f'####                      unchanged leftx, startx:{start_leftX}')
            self.unchanged_leftx = True
            self.l_count = 0
            self.current_leftX = start_leftX
        else:
            self.unchanged_leftx = False

        if self.r_count > 1: 
            # print(f'####                      unchanged rightx, startx:{start_rightX}')
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
        # self.get_logger().info(f'lane encoder count publising:{msg.data}')
        print(' ')


    def sliding_window(self, current_leftX: int, current_rightX: int) -> float:
        """start_x값을 가지고 슬라이딩 윈도우로 center 점 추출 및 steer 계산
        Args
            current_leftX: sliding window 시작 left x
            current_rightX: sliding window 시작 right x
        Return
            rounded_steer: 최종 steer값 """
        left_firstX = current_leftX
        right_firstX = current_rightX
        num_windows = 12  # 윈도우 개수
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
            if len(right_inds) > min_num_pixel:
                valid_right_win_count += 1
                current_rightX = int(np.percentile(nonzerox[right_inds], 10))

            left_x.append(current_leftX)
            right_x.append(current_rightX)

        if 0 < valid_left_win_count < 3:
            print('왼쪽무시')
            current_leftX = -1
            win_left_lane = [np.array([], dtype=int) for _ in range(num_windows)]
            left_x = [-1] * 12
            cv2.putText(self.vis_img, '왼쪽 값 무시!!', (20, 105), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (200, 0, 0), 1, cv2.LINE_AA)
        if 0 < valid_right_win_count < 3:
            print('오른쪽무시')
            current_rightX = 641
            win_right_lane = [np.array([], dtype=int) for _ in range(num_windows)]
            right_x = [641] * 12
            cv2.putText(self.vis_img, '오른쪽 값 무시!!', (20, 135), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (200, 0, 0), 1, cv2.LINE_AA)

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
        ''' center points(list)를 가지고 encoder count 계산
        Args
            current_leftX: 가장 위의 window의 leftX값
            current_rightX: 가장 위의 window의 rightX값(max steer 판단에 사용)'''

		# 파라미터 :  
        conditions = {
            (right_last_X < 320 and abs(right_last_X-right_first_X) > 190) or (right_last_X < 320 and 320+150 > right_first_X and abs(right_last_X-right_first_X) > 160): 1, # no left lane, left steer max 
            (320 < left_last_X and abs(left_first_X - left_last_X) > 190) or (320 < left_last_X and 320-150 < left_first_X and abs(left_first_X - left_last_X) > 160): 2  # no right lane, right steer max
            }
        print(f'l diff:{abs(left_first_X - left_last_X)}')
        print(f'r diff:{abs(right_last_X-right_first_X)}')
        self.mode = conditions.get(True, 0) 
        
        a = 13 # mean_center_x weight
        b = 7 # x_diff weight
        ''' default: 16,8 
        	curve: 13 7 '''

        # steer 계산
        center_points = np.array(center_points)
        mean_center = np.mean(center_points[:,0])

        steer = (mean_center - self.w//2)/a

        first_center = center_points[2][0] 
        last_center = center_points[0][0]

        diff = first_center - last_center

        # 파라미터
        if abs(diff) > 1.2 and steer * diff > 0: # 상쇄되는 거 방지!
            steer = steer + diff/b # parameter 조정

        steer = np.clip(steer, -11.0, 11.0)

        rounded_steer = -round(steer, 1) # 좌회전이 +, 우회전이 -

        real_mode_map = {1: 11.0, 2: -11.0} # 1:left max steer, 2:right max steer
        rounded_steer = real_mode_map.get(self.mode, -round(steer, 1))

        if self.mode == 1: # and current_rightX < 320+maxsteer_threshold :
            # self.get_logger().info(f'mode: {self.mode}, left steer')
            cv2.putText(self.real_img, f'self.mode:{self.mode}', (20,105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,0,0), 1, cv2.LINE_AA)

        elif self.mode == 2: # and 320-maxsteer_threshold < current_leftX:
            # self.get_logger().info(f'mode: {self.mode}, right steer')
            cv2.putText(self.real_img, f'self.mode:{self.mode}', (20,105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,0,0), 1, cv2.LINE_AA) # 삭제

        self.mode = 0  # mode 리셋
        
        if rounded_steer == 0:
            direction = 'straight'
        elif rounded_steer < 0:
            direction = 'right'
        else:
            direction = 'left' # 삭제 
        
        print(f'steer: {rounded_steer}, diff: {diff/b}, x:{round(((mean_center-320)/a),1)}')
        cv2.putText(self.real_img, f'x:{round(((mean_center-320)/16), 1)}, diff: {diff/a}', (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (20,20,120), 1, cv2.LINE_AA)
        cv2.putText(self.real_img, f'{direction}', (20,65), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (20,20,120), 2, cv2.LINE_AA) # 삭제
        cv2.putText(self.real_img, f'{int(rounded_steer)}', (160,95), cv2.FONT_HERSHEY_TRIPLEX, 2.5,  (20,20,120), 2, cv2.LINE_AA) 
        
        return rounded_steer # -11~11 float값

    def brighten_high_pixels(self, frame, delta=60, thresh=55): # 90, 60
        """
        frame : BGR 이미지 (uint8)
        delta : 밝기를 얼마나 올릴지 (0~255)
        thresh: RGB 평균 임계값
        """
        b, g, r = cv2.split(frame.astype(np.int16))
        mean_rgb = ((r + g + b) // 3).astype(np.uint8)
        mask = mean_rgb >= thresh         

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[...,2][mask] = np.clip(hsv[...,2][mask] + delta, 0, 255)

        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return boosted
    
    def remove_sloped_and_small(self, canny_img: np.ndarray, min_num):

        h, w = canny_img.shape
        mid_x = w // 2

        cleaned = canny_img.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        for L in range(1, num_labels):
            ys, xs = np.where(labels == L)
            cx = xs.mean()
            pts = np.column_stack([xs, ys]).astype(np.float32)
            vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            angle_rad = np.arctan2(vy, vx)
            angle_deg = np.degrees(angle_rad)

            # 수평 라인 판정
            near_horiz = abs(angle_deg) < 20 or abs(abs(angle_deg) - 180) < 20

            remove = near_horiz

            if remove:
                cleaned[labels == L] = 0

        # 너무 작은 컴포넌트 제거 
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        for L in range(1, num_labels):
            if stats[L, cv2.CC_STAT_AREA] < min_num:
                cleaned[labels == L] = 0

        return cleaned


    ########### main loop ##########

    def load_video(self):
        video = cv2.VideoCapture(self.video_path)

        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        while video.isOpened():
            ret, img = video.read()

            if not ret:
                print('***** no frame *****')
                break
            img = cv2.resize(img, (640,360))
            self.real_img = img
            self.h, self.w, _ = self.real_img.shape

            '''default''' # 그림자 없을 때
            # img = cv2.cvtColor(self.real_img, cv2.COLOR_BGR2GRAY) 
            # img = cv2.medianBlur(img, 5)
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img = cv2.Canny(img, 210, 250) 

            ''' jeju 오후 ver, 그림자차이 + 빛 반사'''
            # img = cv2.cvtColor(self.real_img, cv2.COLOR_BGR2GRAY) 
            # # img = cv2.medianBlur(img, 5)
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img = cv2.Canny(img, 200, 230) # 230 250 default
            # img = self.remove_sloped_and_small(img, 10)

            ''' jeju default '''
            img = cv2.cvtColor(self.real_img, cv2.COLOR_BGR2GRAY) 
            # img = cv2.medianBlur(img, 5)
            # img = cv2.GaussianBlur(img, (5,5), 0)
            img = cv2.Canny(img, 50, 100) # 250 270 정도로 하면 될 듯?, 230 250 default
            # img = self.remove_sloped_and_small(img ,60)

            cv2.imshow('canny', img)

            h = img.shape[0]
            w = img.shape[1]

            src = np.float32([[120,270], 
                        [w-120,270],
                        [0,h-30],
                        [w,h-30]]) 
            
            for point in src:
                x, y = int(point[0]), int(point[1]) 
                cv2.circle(self.real_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1) # 시각화용
            
            dst = np.float32([[0,0],
                            [w,0],
                            [0,h],
                            [w,h]])

            M = cv2.getPerspectiveTransform(src, dst)
            self.Minv = cv2.getPerspectiveTransform(dst, src) # warp한 이미지를 다시 바꾸기 위한 행렬
            self.warp_img = cv2.warpPerspective(img, M, (w,h))

            cv2.line(self.real_img, (0,self.real_img.shape[0]//2), (self.real_img.shape[1],self.real_img.shape[0]//2), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(self.real_img, (self.real_img.shape[1]//2, 0), (self.real_img.shape[1]//2, self.real_img.shape[0]), (255,255,255), 1, cv2.LINE_AA) # 중간에 라인 생성

            self.histogram() 

            # self.vis_img = cv2.cvtColor(self.vis_img, cv2.COLOR_GRAY2BGR) 
            combined_img = np.hstack((self.vis_img, self.real_img))
            cv2.imshow('result', combined_img)

            if cv2.waitKey(33) == ord('q'): 
                break

        video.release()
        cv2.destroyAllWindows()
            
def main(args=None):
    rospy.init_node('CvEncoderCount', anonymous=True)
    try:
        detector = CvLanedetection()
        detector.load_video()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()