import os
import json

''' json 형식(labelme) -> txt 형식(YOLO)
파일 내에서 json 찾은 후 txt로 바꿔줌(json은 삭제)'''

class_map = {
    "green": 1,  # 필요 시 추가
    "Green": 1,
    "red" : 0,
    "Red" : 0,
    "Invisible" : 2,
    "invisible" : 2
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
            print(f'class 다름, {label}')
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
folder = '/home/leejunmi/Invisible'

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



############################# image, label 분리
# import os
# import shutil

# # 기준 폴더
# base_dir = '/home/leejunmi/new_traffic/train'  # 여기를 실제 폴더 경로로 수정

# # 이동할 폴더 경로
# image_dir = os.path.join(base_dir, 'images')
# label_dir = os.path.join(base_dir, 'labels')

# # 없으면 생성
# os.makedirs(image_dir, exist_ok=True)
# os.makedirs(label_dir, exist_ok=True)

# # 파일 분류
# for file in os.listdir(base_dir):
#     file_path = os.path.join(base_dir, file)
#     if os.path.isfile(file_path):
#         if file.lower().endswith('.jpg'):
#             shutil.move(file_path, os.path.join(image_dir, file))
#         elif file.lower().endswith('.txt'):
#             shutil.move(file_path, os.path.join(label_dir, file))

# print("✔ 분류 완료")


################ 파일 수
# import os

# def count_all_files(directory):
#     total_files = 0
#     for root, dirs, files in os.walk(directory):
#         total_files += len(files)
#     return total_files

# # 사용 예시
# path = "/home/leejunmi/only_leftgreen/dataset/images/val"  # 원하는 디렉토리 경로로 변경
# print("총 파일 수:", count_all_files(path))


######################33
# import os
# import json
# import shutil

# def copy_json_and_jpg_with_invisible(src_dir, dst_dir):
#     os.makedirs(dst_dir, exist_ok=True)

#     for root, _, files in os.walk(src_dir):
#         for file in files:
#             if file.endswith(".json"):
#                 json_path = os.path.join(root, file)
#                 try:
#                     with open(json_path, 'r') as f:
#                         data = json.load(f)
#                         if any(shape.get('label') == 'Invisible' for shape in data.get('shapes', [])):
#                             # 파일명 (확장자 제외)
#                             base_name = os.path.splitext(file)[0]

#                             # 원본 경로들
#                             jpg_path = os.path.join(root, base_name + ".jpg")
#                             json_dst = os.path.join(dst_dir, file)
#                             jpg_dst = os.path.join(dst_dir, base_name + ".jpg")

#                             # 복사
#                             shutil.copy2(json_path, json_dst)
#                             if os.path.exists(jpg_path):
#                                 shutil.copy2(jpg_path, jpg_dst)
#                             else:
#                                 print(f"[경고] JPG 없음: {jpg_path}")

#                 except Exception as e:
#                     print(f"[에러] {json_path} 읽기 실패: {e}")

# # 사용 예
# source_folder = "/home/leejunmi/Traffic_image"
# destination_folder = "/home/leejunmi/Invisible"
# copy_json_and_jpg_with_invisible(source_folder, destination_folder)


