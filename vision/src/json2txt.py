import os
import json

''' json 형식(labelme) -> txt 형식(YOLO)
파일 내에서 json 찾은 후 txt로 바꿔줌(json은 삭제)'''

class_map = {
    "left_green": 3,  # 필요 시 추가
}

def labelme_to_yolo(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    w = data['imageWidth']
    h = data['imageHeight']

    lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            continue
        cls_id = class_map[label]

        pt1, pt2 = shape['points']
        x1, y1 = pt1
        x2, y2 = pt2

        x_center = (x1 + x2) / 2.0 / w
        y_center = (y1 + y2) / 2.0 / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 저장할 txt 경로
    txt_path = os.path.splitext(json_path)[0] + '.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))

    # json 삭제
    os.remove(json_path)
    print(f"✅ 변환 및 삭제 완료: {os.path.basename(json_path)} → {os.path.basename(txt_path)}")

# === 실행: 지정된 폴더 내 전체 json 변환 ===
folder = '/home/leejunmi/only_leftgreen'

for file in os.listdir(folder):
    if file.endswith('.json'):
        json_file = os.path.join(folder, file)
        try:
            labelme_to_yolo(json_file)
        except Exception as e:
            print(f"❌ 오류 ({file}): {e}")



############### label 일치하는지 확인
# import cv2

# img = cv2.imread('/home/leejunmi/only_leftgreen/morai_traffic_00600.jpg')
# h, w = img.shape[:2]

# with open('/home/leejunmi/only_leftgreen/morai_traffic_00600.txt', 'r') as f:
#     for line in f:
#         cls, x, y, bw, bh = map(float, line.split())
#         x1 = int((x - bw / 2) * w)
#         y1 = int((y - bh / 2) * h)
#         x2 = int((x + bw / 2) * w)
#         y2 = int((y + bh / 2) * h)

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#         cv2.putText(img, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

# cv2.imshow("check", img)
# cv2.waitKey(0)
