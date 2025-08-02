#from ultralytics import YOLO
import cv2
import time
import numpy as np
# ''' 3구일때 처리 필요함 '''

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
            
        
        # Convert ROI to HSV color space.
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.hsv1 = cv2.cvtColor(self.roi1, cv2.COLOR_BGR2HSV)
        self.hsv2 = cv2.cvtColor(self.roi2, cv2.COLOR_BGR2HSV)
        self.hsv3 = cv2.cvtColor(self.roi3, cv2.COLOR_BGR2HSV)
        self.hsv4 = cv2.cvtColor(self.roi4, cv2.COLOR_BGR2HSV)
        #cv2.imshow("test",self.hsv)
        # #cv2.imshow(np.hstack((self.hsv1,self.hsv2,self.hsv3,self.hsv4)))
        
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
        # 여기서 바로 True False 보내도됨

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