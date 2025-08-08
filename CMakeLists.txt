cmake_minimum_required(VERSION 3.0.2)
project(vision)

# ----------  의존 패키지 ----------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)
find_package(OpenCV REQUIRED)

# ----------  catkin 패키지 선언 ----------
# add_executable을 내보내기 위해 필요함
catkin_package()

# ----------  include 디렉토리 ----------
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# ----------  노드 빌드 ----------
add_executable(traffic_wonju src/traffic_wonju.cpp)
add_dependencies(traffic_wonju ${catkin_EXPORTED_TARGETS})
target_link_libraries(traffic_wonju
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Python 스크립트
catkin_install_python(
  PROGRAMS src/image.py src/image_sub.py src/traffic_all_ver.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS traffic_wonju
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

