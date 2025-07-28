# import os
# from collections import defaultdict

# # 분석할 폴더 경로 지정
# folder_path = '/home/leejunmi/YOLODataset/labels/val'  # TODO: 여기에 실제 경로 입력

# # 전체 클래스 카운트 저장용
# total_class_counts = defaultdict(int)

# # 폴더 내 모든 .txt 파일 반복
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)

#         with open(file_path, 'r') as f:
#             for line in f:
#                 if not line.strip():
#                     continue  # 빈 줄 skip

#                 class_id = line.strip().split()[0]

#                 if class_id.isdigit():
#                     total_class_counts[int(class_id)] += 1

# # 출력
# for class_id in sorted(total_class_counts):
#     print(f"Class {class_id}: {total_class_counts[class_id]}개")

#######################################################
# import os
# ㄴ
# folder_path = '/home/leejunmi/YOLODataset/labels/train'  # TODO: 여기에 경로 입력

# # 해당 디렉토리 내 파일 개수 세기
# file_count = sum(1 for f in os.listdir(folder_path)
#                  if os.path.isfile(os.path.join(folder_path, f)))

# print(f"📁 {folder_path} 안의 파일 개수: {file_count}개")


######################################################33


# import os
# import glob

# # 사용자 홈 디렉토리 경로
# home_dir = os.path.expanduser('~')

# # traffic_morai로 시작하는 모든 파일 경로 검색
# pattern = os.path.join(home_dir, 'morai_traffic*')
# files_to_delete = glob.glob(pattern)

# # 삭제
# for file_path in files_to_delete:
#     try:
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#             print(f'✅ 삭제됨: {file_path}')
#         else:
#             print(f'⚠️ 파일 아님 (건너뜀): {file_path}')
#     except Exception as e:
#         print(f'❌ 오류 발생: {file_path}, {e}')

import os
import json
import shutil

# 경로 설정
base_dir = '/home/leejunmi/Traffic_image'
target_dir = '/home/leejunmi/only_leftgreen'

# 폴더 없으면 생성
os.makedirs(target_dir, exist_ok=True)

# 순회
for file in os.listdir(base_dir):
    if file.endswith('.json'):
        json_path = os.path.join(base_dir, file)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

                # "left_green" 포함 여부 확인
                if any(shape.get('label') == 'left_green' for shape in data.get('shapes', [])):
                    # .json 복사
                    shutil.copy(json_path, os.path.join(target_dir, file))

                    # .jpg 복사
                    jpg_name = file.replace('.json', '.jpg')
                    jpg_path = os.path.join(base_dir, jpg_name)
                    if os.path.exists(jpg_path):
                        shutil.copy(jpg_path, os.path.join(target_dir, jpg_name))

                    print(f"✅ 복사됨: {file} + {jpg_name}")

        except Exception as e:
            print(f"❌ 오류 발생 - {file}: {e}")



