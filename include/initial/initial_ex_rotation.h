#pragma once

#include <vector>
#include "parameters.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;

// #include <ros/console.h>

class InitialEXRotation
{
public:
    InitialEXRotation();
    bool CalibrationExRotation(std::vector<std::pair<Vector3d, Vector3d>> corres, 
                               Quaterniond delta_q_imu, 
                               Eigen::Matrix3d &calib_ric_result);

private:
    Matrix3d solveRelativeR(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres);

    double testTriangulation(const std::vector<cv::Point2f> &l,
                             const std::vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    
private:
    /*!< @brief 输入当前类的图像数量 */
    int                 frame_count;
    /*!< @brief 输入当前类的图像对之间的旋转变换 */
    std::vector<Matrix3d>    Rc;
    /*!< @brief 输入当前类的IMU陀螺仪积分值 */
    std::vector<Matrix3d>    Rimu;
    /*!< @brief 使用IMU陀螺仪积分值以及外参表示的图像对之间的旋转变换 */
    std::vector<Matrix3d>    Rc_g;
    /*!< @brief 输出成员变量：IMU-Camera之间的旋转变换 */
    Eigen::Matrix3d          ric;
};
