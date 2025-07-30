#from ultralytics import YOLO
import cv2
import time
import numpy as np

# model = YOLO("best.pt")
# model.to('cuda')
# # 비디오 파일 경로를 적절하게 변경하세요.
# video_path = "/home/macaron/바탕화면/MORAI_V/13_20-21-04.avi"
# confidence_limit = 0.6

# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # ROI 좌표 (왼쪽 위와 오른쪽 아래)
# roi_coordinates = ((0, 0), (1980, 300))

class TrafficLightDetector:
    def __init__(self, image, x, y, w, h):
        print(image.shape)
        w,h = image.shape[1], image.shape[0]
        # Extract region of interest (ROI) from the image based on the rectangle
        # self.roi = image[y:y+h, x:x+w]
        # self.roi1 = image[y:y+h, x+w//12:x+w*3//12]
        # self.roi2 = image[y:y+h, x+w*4//12:x+w*6//12]
        # self.roi3 = image[y:y+h, x+w*6//12:x+w*8//12]
        # self.roi4 = image[y:y+h, x+w*9//12:x+w]
        self.roi = image[:,:]
        self.roi1 = image[:,:w*3//12]
        self.roi2 = image[:,w*3//12:w*2//4]
        self.roi3 = image[:,w*2//4:w*3//4]
        self.roi4 = image[:,w*3//4:w*4//4]
        cv2.imshow("test",self.roi4)
        cv2.waitKey(0)
        
        # Convert ROI to HSV color space.
        self.hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.hsv1 = cv2.cvtColor(self.roi1, cv2.COLOR_BGR2HSV)
        self.hsv2 = cv2.cvtColor(self.roi2, cv2.COLOR_BGR2HSV)
        self.hsv3 = cv2.cvtColor(self.roi3, cv2.COLOR_BGR2HSV)
        self.hsv4 = cv2.cvtColor(self.roi4, cv2.COLOR_BGR2HSV)
        #cv2.imshow("test",self.hsv)
        # cv2.imshow(np.hstack((self.hsv1,self.hsv2,self.hsv3,self.hsv4)))
        # cv2.waitKey(0)
        
    
        # Define HSV range for red color
        self.red_lower = np.array([0, 120, 70])
        self.red_upper = np.array([10, 255, 255])
        # Define HSV range for yellow color
        self.yellow_lower = np.array([20, 100, 100])  # 수정된 값
        self.yellow_upper = np.array([35, 255, 255])  # 수정된 값 # 30 -> 35

        self.green_lower = np.array([35, 100, 100])  # 수정된 값
        self.green_upper = np.array([100, 255, 255])  # 수정된 값

        # 빨간색, 노란색 범위에 해당하는 부분을 필터링합니다.
        self.red_mask = cv2.inRange(self.hsv1, self.red_lower, self.red_upper)
        self.yellow_mask = cv2.inRange(self.hsv2, self.yellow_lower, self.yellow_upper)
        # 초록색 범위에 해당하는 부분을 필터링합니다.
        self.green_left_mask = cv2.inRange(self.hsv3, self.green_lower, self.green_upper)
        self.green_mask = cv2.inRange(self.hsv4, self.green_lower, self.green_upper)
        print(self.yellow_mask)
        
    def detect(self):
        print(cv2.countNonZero(self.yellow_mask))
        #red detect
        if cv2.countNonZero(self.red_mask) > 3:
            print('****')
            if cv2.countNonZero(self.green_left_mask) > 0:
                return 'red and green'
            elif cv2.countNonZero(self.yellow_mask) > 0:
                return 'red and yellow'
            else:
                return 'red'
        #orange detect
        elif cv2.countNonZero(self.yellow_mask) > 0:
            return 'yellow'
         
        elif cv2.countNonZero(self.green_mask) > 0:
            if cv2.countNonZero(self.green_left_mask) > 0:
                return 'all_green'
            else:
                return 'straight_green'
        elif cv2.countNonZero(self.green_left_mask) > 0:
            return 'left_green'
        
        else:
            return 'None'

img = cv2.imread('/home/leejunmi/Y.png')

traffic = TrafficLightDetector(img,1,1,1,1)
result = traffic.detect()
print(result)

# def detect_video():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame is not None:
#         # ROI 적용
#         roi_image = frame[roi_coordinates[0][1]:roi_coordinates[1][1], roi_coordinates[0][0]:roi_coordinates[1][0]]
       
#         t = time.time()
#         # YOLO 모델을 사용하여 결과를 얻습니다.
#         results = model(roi_image)
#         s = time.time() - t
#         print(s)
#         # 예측된 객체의 바운딩 박스 정보를 출력합니다.
#         boxes = results[0].boxes.cpu().numpy().data
        
#         for box in boxes:
#             label = results[0].names[int(box[5])]
#             if box[4] > confidence_limit:
#                 left = int(box[0])
#                 top = int(box[1])
#                 right = int(box[2])
#                 bottom = int(box[3])
#                 label = results[0].names[int(box[5])]

                
#             elif box[4] < confidence_limit :
#                 # Determine the color of the detected traffic light
#                 traffic = TrafficLightDetector(frame, left, top, abs(right-left), abs(bottom - top))
#                 sign = traffic.detect()
#                 # Overlay the detected traffic light status on the main image
#                 font = cv2.FONT_HERSHEY_TRIPLEX
#                 font_scale = 1  
#                 text_color = (0, 0, 255)  # Default color for text
                    
#                 if sign == 'left_green':
#                     label = "left_green"       
#                     text_color = (0, 255, 0)  # green

                
#                 elif sign == 'straight_green':
#                     label = "straight_green"
#                     text_color = (0, 255, 0)  # green
                    
#                 elif sign == 'all_green':
#                     message = 'all_green'
#                     text_color = (0, 255, 0)  # green
#                 elif sign == 'orange':
#                     label  = "orange"
#                     text_color = (0, 255, 255)  # yellow

                
#                 elif sign == 'red':
#                     label  = "red"
#                     text_color = (0, 0, 255)  # red
               
#                 else:
#                     label = "None"
#                     text_color = (0, 255, 0)  # Green
#                 print("{}".format(label))
#                 # Return the modified image and detected color
#                 text_size = cv2.getTextSize(message, font, font_scale-0.5, 2)[0]
#                 text_width, text_height = text_size[0], text_size[1]
#                 text_x = max(0, (left + right - text_width) // 2) # left에서 살짝 떨어진 거리에 표시
#                 text_y = max(0, bottom + text_height + 10)  # 10은 여유 공간입니다.
#                 cv2.putText(frame, label, (text_x, text_y), font, font_scale-0.5, text_color)
#                 cv2.rectangle(frame, (left, bottom), (right, top), (0, 255, 0), 2)
#                 # 결과 이미지 출력
                
#             else:
#                 frame = results[0].plot()
 
#                     # cv2.imshow("Traffic Light Detection", results)
                
#             cv2.imshow("plot", frame)
#             # 'q' 키를 눌러 창을 닫습니다.
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
             

# # # 비디오 캡처 해제 및 창 종료
# # cap.release()
# # cv2.destroyAllWindows()