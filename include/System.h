#pragma once

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <condition_variable>

#include <pangolin/pangolin.h>
#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"

/* IMU信息 */
struct IMU_MSG
{
    double header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

/* 图像信息 */    
struct IMG_MSG {
    double header;
    std::vector<Vector3d> points;
    std::vector<int> id_of_point;
    std::vector<float> u_of_point;
    std::vector<float> v_of_point;
    std::vector<float> velocity_x_of_point;
    std::vector<float> velocity_y_of_point;
};
typedef std::shared_ptr<IMG_MSG const> ImgConstPtr;
    
class System
{
public:
	/*!
	*  @brief VIO系统构造函数
	*  @param[in]	sConfig_files	系统配置文件路径
	*/
    System(std::string sConfig_files);

	/*!
	*  @brief VIO系统析构函数
	*/
    ~System();

	/*!
	*  @brief 对传入VIO系统的图像进行处理
	*  @param[in]	dStampSec	输入图像时间戳
	*  @param[in]	img			输入图像数据
	*/
    void PubImageData(double dStampSec, cv::Mat &img);

	/*!
	*  @brief 对传入VIO系统的图像进行处理：模拟数据
	*  @param[in]	dStampSec		输入图像时间戳
	*  @param[in]	featurePoints	输入图像特征
	*/
	void PubImageData(double dStampSec, const std::vector<cv::Point2f>& featurePoints);

	/*!
	*  @brief 对传入VIO系统的IMU数据进行处理
	*  @param[in]	dStamp	输入IMU数据时间戳
	*  @param[in]	vGyr	输入IMU数据角速度
	*  @param[in]	vAcc	输入IMU数据加速度
	*/
    void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
					const Eigen::Vector3d &vAcc);

    // thread: visual-inertial odometry
    void ProcessBackEnd();
    void Draw();
    
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;

#ifdef __APPLE__
    void InitDrawGL(); 
    void DrawGLFrame();
#endif

private:

    //feature tracker
    std::vector<uchar> r_status;
    std::vector<float> r_err;
    // std::queue<ImageConstPtr> img_buf;

    // ros::Publisher pub_img, pub_match;
    // ros::Publisher pub_restart;

    FeatureTracker trackerData[NUM_OF_CAM];
    double first_image_time;
    int pub_count = 1;
    bool first_image_flag = true;
    double last_image_time = 0;
    bool init_pub = 0;

    //estimator
    Estimator estimator;

    std::condition_variable con;
    double current_time = -1;
    std::queue<ImuConstPtr> imu_buf;
    std::queue<ImgConstPtr> feature_buf;
    // std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;
    std::mutex m_state;
    std::mutex i_buf;
    std::mutex m_estimator;

    double latest_time;
    Eigen::Vector3d tmp_P;
    Eigen::Quaterniond tmp_Q;
    Eigen::Vector3d tmp_V;
    Eigen::Vector3d tmp_Ba;
    Eigen::Vector3d tmp_Bg;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;
    bool init_feature = 0;
    bool init_imu = 1;
    double last_imu_t = -1;
    std::ofstream ofs_pose;
	pangolin::OpenGlMatrix currentTwc;
    std::vector<Eigen::Vector3d> vPath_to_draw;
	std::vector<Eigen::Vector3d> vLandMark_to_draw;
    bool bStart_backend;

	/*!
	*  @brief VIO系统中将IMU数据与相机数据对齐：获取时间对齐的相机与IMU测量数据
	*  @return	std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>>	对齐后的IMU数据与相机数据
	*/
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
};