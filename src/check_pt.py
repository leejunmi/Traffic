import cv2
from ultralytics import YOLO

model = YOLO('/home/leejunmi/catkin_ws/src/vision/src/best_train_custom2.pt')
# model = YOLO('/home/leejunmi/catkin_ws/src/vision/src/best_0729.pt')  


# VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/no_gps_obstacle#5.avi"
VIDEO_PATH = "/home/leejunmi/catkin_ws/src/vision/src/output1.avi" 
cap = cv2.VideoCapture(VIDEO_PATH)  



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0] 

    boxes = [b for b in results.boxes if (b.xyxy[0][2] - b.xyxy[0][0]) >= (b.xyxy[0][3] - b.xyxy[0][1])] # 세로가 더 긴 박스 제거
    if not boxes:
        print('no boxes')
        cv2.imshow('YOLO Traffic Detection', frame)


    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        cls = int(box.cls[0])  # 클래스 인덱스
        conf = float(box.conf[0])  # 신뢰도
        label = f'{model.names[cls]} {conf:.2f}'
        print(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


############################### pt 내의 클래스를 확인 
# from ultralytics import YOLO

# model = YOLO('/home/leejunmi/catkin_ws/src/vision/src/only_traffic_lane.pt') 
# print(model.names)  
