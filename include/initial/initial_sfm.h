#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Dense>

#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;

/*!
*  @brief 滑动窗口中所有的路标点：每个路标点由多个连续图像观测到
*  @param	state		特征点的状态：是否被三角化
*  @param	id			特征点的ID
*  @param	observation	所有观测到该特征点的图像帧ID与2D像素坐标
*  @param	position	特征点3D坐标
*  @param	depth		特征点深度信息
*/
struct SFMFeature
{
    bool state;
    int id;
    std::vector<std::pair<int,Vector2d>> observation;
    double position[3];
    double depth;
};

/*!
*  @brief 使用ceres进行重投影误差优化时重载的函数操作
*/
struct ReprojectionError3D
{
	/*!
	*  @brief 重投影误差计算构造函数
	*  @param[in]	observed_u	像素坐标u方向观测值
	*  @param[in]	observed_v	像素坐标v方向观测值
	*/
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v){}

	/*!
	*  @brief 通过重载()运算符，计算优化问题残差
	*  @param[in]	camera_R	世界系到相机系下的旋转
	*  @param[in]	camera_T	世界系到相机系下的平移
	*  @param[in]	point		世界系下的路标点
	*  @param[in]	residuals	重投影误差
	*  @return		bool		残差是否计算成功
	*/
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	/*!
	*  @brief 优化问题中损失函数构造函数
	*  @param[in]	observed_x	像素坐标u方向观测值
	*  @param[in]	observed_y	像素坐标v方向观测值
	*  @return		ceres::CostFunction*	损失函数
	*/
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	/*!< @brief 像素坐标u方向观测值 */
	double observed_u;
	/*!< @brief 像素坐标v方向观测值 */
	double observed_v;
};

class GlobalSFM
{
public:
	/*!
	*  @brief 视觉初始化全局SFM构造函数
	*/
	GlobalSFM();

	/*!
	*  @brief 纯视觉SFM：求解滑窗中所有图像帧相对于第l帧的位姿以及三角化特征点坐标
	*  @param[in]	frame_num	窗口总帧数:frame_count+1
	*  @param[out]	q			窗口内图像帧的旋转四元数：相对于第l帧
	*  @param[out]	T			窗口内图像帧的平移向量：相对于第l帧
	*  @param[in]	l			窗口内定义的第l帧
	*  @param[in]	relative_R	当前帧到第l帧的旋转矩阵
	*  @param[in]	relative_T	当前帧到第l帧的平移向量
	*  @param[in]	sfm_f		滑动窗口内的所有特征点
	*  @param[out]	sfm_tracked_points	所有在SFM中三角化的特征点ID和坐标
	*  @return		bool		纯视觉SFM是否求解成功
	*/
	bool construct(int& frame_num, std::vector<Eigen::Quaterniond>& q, std::vector<Eigen::Vector3d>& T, 
			  int& l, const Eigen::Matrix3d& relative_R, const Eigen::Vector3d& relative_T,
			  std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points);

private:
	/*!
	*  @brief 通过PnP方法得到当前帧相对于第l帧的位姿
	*  @param[in/out]	R_initial	当前所求图像帧姿态初始值
	*  @param[in/out]	P_initial	当前所求图像帧位置初始值
	*  @param[in]		i			当前所求图像帧为滑窗中第i帧
	*  @param[in]		sfm_f		滑窗中的所有路标点	
	*/
	bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, 
						std::vector<SFMFeature> &sfm_f);

	/*!
	*  @brief 通过三角化方法恢复3D世界坐标
	*  @param[in]	Pose0		第0帧相机世界姿态
	*  @param[in]	Pose1		第1帧相机世界姿态
	*  @param[in]	point0		特征点在第0帧相机的归一化相机坐标	
	*  @param[in]	point1		特征点在第1帧相机的归一化相机坐标
	*  @param[out]	point_3d	特征点三角化得到的对应世界坐标
	*/
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);

	/*!
	*  @brief 通过三角化方法求解两帧图像匹配点的世界坐标
	*  @param[in]		frame0		第0帧图像帧的ID
	*  @param[in]		frame1		第1帧图像帧的ID
	*  @param[in]		Pose0		第0帧相机世界姿态
	*  @param[in]		Pose1		第1帧相机世界姿态
	*  @param[in/out]	point_3d	特征点序列
	*/
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  std::vector<SFMFeature> &sfm_f);
	/*!< @brief 特征点数量 */
	int feature_num;
};