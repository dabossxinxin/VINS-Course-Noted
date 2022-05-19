#pragma once

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace myslam {
namespace backend {

/**
* 视觉重投影误差边：为三元边，相连顶点顺序为
* 路标点的逆深度InverseDepth、第一次观测到该路标的相机位姿T_World_From_Body1、
* 最后一个观测到该路标点的相机位姿T_World_From_Body2，相机与IMU之间的姿态固定
*/
class EdgeReprojectionICFixed :public Edge {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	
	/*!
	*  @brief 视觉重投影边构造函数
	*  @param[in]	pts_i_	首次观测的归一化相机坐标
	*  @param[in]	pts_j_	末次观测的归一化相机坐标
	*/
	EdgeReprojectionICFixed(const Vec3& pts_i, const Vec3& pts_j)
		:Edge(2, 3, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose"}) {
		pts_i_ = pts_i;
		pts_j_ = pts_j;
	}

	/*!
	*  @brief 获取当前边的类型信息
	*  @return	std::strin	边的类型信息
	*/
	virtual std::string TypeInfo() const override {
		return "EdgeReprojectionICFixed";
	}

	/*!
	*  @brief 计算当前边的残差
	*/
	virtual void ComputeResidual() override;

	/*!
	*  @brief 计算当前边对链接顶点的雅可比
	*/
	virtual void ComputeJacobians() override;

	/*!
	*  @brief 设置当前相机与IMU之间的外参
	*  @param[in]	qic_	相机与IMU之间的姿态差
	*  @param[in]	tic_	相机与IMU之间的位置差
	*/
	void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);
	
private:
	/*!< @brief 相机与IMU之间的姿态差 */
	Qd		qic;
	/*!< @brief 相机与IMU之间的位置差 */
	Vec3	tic;

	/*!< @brief 路标点在i相机系的归一化相机坐标 */
	Vec3 pts_i_;
	/*!< @brief 路标点在j相机系的归一化相机坐标 */
	Vec3 pts_j_;
};

 /**
 * 视觉重投影误差边：为四元边，相连顶点顺序为
 * 路标点的逆深度InverseDepth、第一次观测到该路标的相机位姿T_World_From_Body1、
 * 最后一个观测到该路标点的相机位姿T_World_From_Body2、相机与IMU之间的位姿变换
 */
class EdgeReprojection : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/*!
	*  @brief 视觉重投影边构造函数
	*  @param[in]	pts_i_	首次观测的归一化相机坐标
	*  @param[in]	pts_j_	末次观测的归一化相机坐标
	*/
    EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"}) {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }

	/*!
	*  @brief 获取当前边的类型信息
	*  @return	std::strin	边的类型信息
	*/
    virtual std::string TypeInfo() const override {
		return "EdgeReprojection";
	}

	/*!
	*  @brief 计算当前边的残差
	*/
    virtual void ComputeResidual() override;

	/*!
	*  @brief 计算当前边对链接顶点的雅可比
	*/
    virtual void ComputeJacobians() override;

	/*!
	*  @brief 设置当前相机与IMU之间的外参
	*  @param[in]	qic_	相机与IMU之间的姿态差
	*  @param[in]	tic_	相机与IMU之间的位置差
	*/
    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
	/*!< @brief 相机与IMU之间的姿态差 */
    Qd		qic;
	/*!< @brief 相机与IMU之间的位置差 */
    Vec3	tic;

	/*!< @brief 世界坐标点在两个不同相机位置的观测值 */
    Vec3 pts_i_, pts_j_;
};

/**
* 视觉重投影误差边：为二元边，相连顶点顺序为
* 路标点的世界坐标XYZ、观测到该路标的相机位姿T_World_From_Body1、
*/
class EdgeReprojectionXYZ : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/*!
	*  @brief 视觉重投影边构造函数
	*  @param[in]	pts_i_	路标点的世界坐标观测值
	*/
    EdgeReprojectionXYZ(const Vec3 &pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        obs_ = pts_i;
    }

	/*!
	*  @brief 获取边的类型信息
	*  @return	std::string	边的类型信息
	*/
    virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }

	/*!
	*  @brief 计算当前边的残差
	*/
    virtual void ComputeResidual() override;

	/*!
	*  @brief 计算当前边对顶点的雅可比
	*/
    virtual void ComputeJacobians() override;

	/*!
	*  @brief 设置当前相机与IMU之间的外参
	*  @param[in]	qic_	相机与IMU之间的姿态差
	*  @param[in]	tic_	相机与IMU之间的位置差
	*/
    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
	/*!< @brief 相机与IMU之间的姿态差 */
    Qd		qic;
	/*!< @brief 相机与IMU之间的位置差 */
    Vec3	tic;

	/*!< @brief 路标点在世界坐标系中的位置观测 */
    Vec3	obs_;
};

 /**
 * 视觉重投影误差边：为一元边，相连顶点为
 * 观测到该路标的相机位姿T_World_From_Body1、
 */
class EdgeReprojectionPoseOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/*!
	*  @brief 视觉重投影边构造函数
	*  @param[in]	landmark_world	路标点的世界坐标观测值
	*  @param[in]	K				相机内参矩阵
	*/
    EdgeReprojectionPoseOnly(const Vec3 &landmark_world, const Mat33 &K) :
        Edge(2, 1, std::vector<std::string>{"VertexPose"}),
        landmark_world_(landmark_world), K_(K) {}

	/*!
	*  @brief 获取当前边的类型信息
	*  @return	std::string	当前边的类型信息
	*/
    virtual std::string TypeInfo() const override { 
		return "EdgeReprojectionPoseOnly";
	}

	/*!
	*  @brief 计算当前边的残差
	*/
    virtual void ComputeResidual() override;

	/*!
	*  @brief 计算当前边相对于顶点的雅可比
	*/
    virtual void ComputeJacobians() override;

private:
	/*!< @brief 路标点在世界坐标系中的位置 */
    Vec3 landmark_world_;
	/*!< @brief 当前相机内参矩阵 */
    Mat33 K_;
};

}
}