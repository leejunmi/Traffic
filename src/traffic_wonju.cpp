// traffic_sign_fov20.cpp  ―  HSV 기반 신호등 감지 (FOV 20)

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Bool.h>
#include <opencv2/opencv.hpp>

//////////////////////////////////////////////////////
// TrafficLightDetector – HSV ROI 분석
//////////////////////////////////////////////////////
class TrafficLightDetector {
public:
    TrafficLightDetector(const cv::Mat& image, ros::Publisher& pub)
        : traffic_pub_(pub)
    {
        cv::imshow("original", image);

        /* 1) ROI & HSV */
        int height = image.rows;
        roi_ = image(cv::Rect(0, 0, image.cols, height - 150));
        cv::cvtColor(roi_, hsv_, cv::COLOR_BGR2HSV);

        /* 2) HSV 범위 */
        red_lower1_   = {  0,120, 70};   red_upper1_  = { 10,255,255};
        red_lower2_   = {170,200,200};   red_upper2_  = {180,255,255};
        yellow_lower_ = { 20,100,100};   yellow_upper_= { 35,255,255};
        green_lower_  = { 35,135,135};   green_upper_ = {100,255,255};

        /* 3) 마스크 */
        cv::inRange(hsv_, red_lower1_,   red_upper1_,   red_mask1_);
        cv::inRange(hsv_, red_lower2_,   red_upper2_,   red_mask2_);
        cv::bitwise_or(red_mask1_, red_mask2_, red_mask_);
        cv::inRange(hsv_, yellow_lower_, yellow_upper_, yellow_mask_);
        cv::inRange(hsv_, green_lower_,  green_upper_,  green_mask_);

        cv::bitwise_or(red_mask_, yellow_mask_, all_mask_);
        cv::bitwise_or(all_mask_, green_mask_,  all_mask_);

        cv::bitwise_and(roi_, roi_, visual_, all_mask_);

        /* 4) 결과 표시 & 퍼블리시 */
        std::string res = analyze();
        cv::putText(visual_, res, {30,30}, cv::FONT_HERSHEY_TRIPLEX,
                    1, {255,255,255}, 1, cv::LINE_AA);

        try { cv::imshow("Traffic Detection", visual_); cv::waitKey(1); }
        catch (...) {}
    }

private:
    std::string analyze()
    {
        cv::Mat labels, stats, centroids;
        int num = cv::connectedComponentsWithStats(all_mask_,
                                                   labels, stats, centroids);

        if (num <= 1) { publish(false); return "Go"; }

        /* 최대 면적 라벨 찾기 */
        int best = 1;
        int best_area = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int i = 2; i < num; ++i) {
            int a = stats.at<int>(i, cv::CC_STAT_AREA);
            if (a > best_area) { best_area = a; best = i; }
        }

        /* 바운딩 박스 */
        int x = stats.at<int>(best, cv::CC_STAT_LEFT);
        int y = stats.at<int>(best, cv::CC_STAT_TOP);
        int w = stats.at<int>(best, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(best, cv::CC_STAT_HEIGHT);
        double cx = centroids.at<double>(best,0);
        double cy = centroids.at<double>(best,1);

        cv::rectangle(visual_, {x,y}, {x+w,y+h}, {255,255,255}, 2);
        cv::circle   (visual_, {static_cast<int>(cx),static_cast<int>(cy)},
                      3, {0,0,255}, -1);

        /* ROI 재분석 */
        cv::Mat roi_hsv = hsv_(cv::Range(y, y+h),
                               cv::Range(x, x+w));

        cv::Mat r1,r2,r_mask,y_mask,g_mask;
        cv::inRange(roi_hsv, red_lower1_, red_upper1_, r1);
        cv::inRange(roi_hsv, red_lower2_, red_upper2_, r2);
        cv::bitwise_or(r1,r2,r_mask);
        cv::inRange(roi_hsv, yellow_lower_, yellow_upper_, y_mask);
        cv::inRange(roi_hsv, green_lower_,  green_upper_,  g_mask);

        int r = cv::countNonZero(r_mask);
        int y_cnt = cv::countNonZero(y_mask);
        int g = cv::countNonZero(g_mask);

        ROS_INFO_STREAM("####  red:" << r << "  yellow:" << y_cnt
                                     << "  green:" << g);

        const int th = 30;
        if (r > th && r < 1500 && r > y_cnt && r > g) {
            publish(true);  return "Stop";
        } else {
            publish(false); return "Go";
        }
    }

    void publish(bool stop) {
        std_msgs::Bool m; m.data = stop;
        traffic_pub_.publish(m);
    }

    /* 멤버 */
    ros::Publisher& traffic_pub_;
    cv::Mat roi_, hsv_, red_mask1_, red_mask2_, red_mask_,
            yellow_mask_, green_mask_, all_mask_, visual_;
    cv::Scalar red_lower1_,red_upper1_,red_lower2_,red_upper2_,
               yellow_lower_,yellow_upper_,green_lower_,green_upper_;
};

//////////////////////////////////////////////////////
// Traffic – 이미지 입력 선택 (비디오 / 토픽)
//////////////////////////////////////////////////////
class Traffic {
public:
    Traffic(bool video_mode, const std::string& video_path)
        : video_mode_(video_mode)
    {
        stop_pub_ = nh_.advertise<std_msgs::Bool>("/stop",1,true);

        if (video_mode_) {
            cap_.open(video_path);
            if (!cap_.isOpened()) {
                ROS_ERROR("Video open failed"); ros::shutdown();
            }
        } else {
            sub_ = nh_.subscribe("/image_jpeg/compressed",1,
                                 &Traffic::imgCallback,this);
        }
    }

    void processFrame(const cv::Mat& frame) {
        TrafficLightDetector detector(frame, stop_pub_);
    }

    /* 비디오 모드 상태 & 캡처 접근자 공개 */
    bool video_mode() const        { return video_mode_; }
    cv::VideoCapture& cap()        { return cap_; }

private:
    /* 콜백 – 토픽 입력 */
    void imgCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        try {
            cv::Mat buf(1, msg->data.size(), CV_8UC1,
                        const_cast<uchar*>(msg->data.data()));
            cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
            processFrame(img);
        } catch (const cv::Exception& e) {
            ROS_ERROR("decode error: %s", e.what());
        }
    }

    ros::NodeHandle   nh_;
    ros::Publisher    stop_pub_;
    ros::Subscriber   sub_;
    cv::VideoCapture  cap_;
    bool              video_mode_;
};

//////////////////////////////////////////////////////
// main
//////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    ros::init(argc, argv, "traffic_sign");

    const bool        VIDEO_MODE = true;          // true → 비디오 파일, false → 이미지 토픽
    const std::string VIDEO_PATH = "/home/leejunmi/VIDEO/output5.avi";

    Traffic traffic(VIDEO_MODE, VIDEO_PATH);

    if (traffic.video_mode()) {
        ros::Rate rate(30);
        while (ros::ok()) {
            cv::Mat frame;
            if (!traffic.cap().read(frame)) {
                ROS_INFO("영상 종료"); break;
            }
            traffic.processFrame(frame);
            ros::spinOnce();
            rate.sleep();
        }
    } else {
        ros::spin();
    }
    return 0;
}
