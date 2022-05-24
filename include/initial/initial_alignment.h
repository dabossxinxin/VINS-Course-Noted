#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <map>

#include "../factor/integration_base.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;


class ImageFrame
{
public:
	/*!
	*  @brief 图像帧默认构造函数
	*/
    ImageFrame(){};

	/*!
	*  @brief 图像帧构造函数
	*  @detail map: (camera_id, [x,y,z,u,v,vx,vy])
	*  @param[in]	_points	图像帧中所有观测到的图像特征
	*  @param[in]	_t		当前图像帧时间戳
	*/
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, 
    double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };

	/*!< @brief 当前图像帧观测到的所有特征 */
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

    /*!< @brief 当前图像帧的时间戳 */
    double                  t;
    /*!< @brief 当前图像帧的姿态 */
    Matrix3d                R;
    /*!< @brief 当前图像帧的位置 */
    Vector3d                T;
    /*!< @brief 当前图像帧的IMU预积分值 */
    IntegrationBase         *pre_integration;
    /*!< @brief 当前图像帧是否为关键帧的标志 */
    bool                    is_key_frame;
};

/*!
*  @brief 系统初始化时将IMU与相机对齐
*  @param[in/out]    all_image_frame   对齐时所有输入图像帧
*  @param[out]       Bgs               IMU陀螺仪偏置
*  @param[out]       g                 相机与IMU之间的标定结果
*  @param[out]       x                 相机与IMU之间的标定结果
*  @return  是否将IMU与相机对齐的标志
*/
bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, 
                        Eigen::Vector3d *Bgs, 
                        Eigen::Vector3d &g, 
                        Eigen::VectorXd &x);