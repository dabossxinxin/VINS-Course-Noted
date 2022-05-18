#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_imu.h"

#include <iostream>

namespace myslam {
namespace backend {

using Sophus::SO3d;

/*!< @brief 重力向量 */
Vec3 EdgeImu::gravity_ = Vec3(0, 0, 9.8);

/**@brief 计算当前边的残差
*/
void EdgeImu::ComputeResidual() {
    /* 获取Pose_i参数 */
    VecX param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi = param_0.head<3>();
    /* 获取speedBias_i参数 */
    VecX param_1 = verticies_[1]->Parameters();
    Vec3 Vi = param_1.head<3>();
    Vec3 Bai = param_1.segment(3, 3);
    Vec3 Bgi = param_1.tail<3>();
    /* 获取Pose_j参数 */
    VecX param_2 = verticies_[2]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj = param_2.head<3>();
    /* 获取speedBias_j参数 */
    VecX param_3 = verticies_[3]->Parameters();
    Vec3 Vj = param_3.head<3>();
    Vec3 Baj = param_3.segment(3, 3);
    Vec3 Bgj = param_3.tail<3>();
    /* 计算当前IMU边的残差 */
    residual_ = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,Pj, Qj, Vj, Baj, Bgj);
    //Mat1515 sqrt_info  = Eigen::LLT< Mat1515 >(pre_integration_->covariance.inverse()).matrixL().transpose();
    /* 设置当前残差的权阵 */
    SetInformation(pre_integration_->covariance.inverse());
}

/**@brief 计算当前边对应优化顶点的雅可比
*/
void EdgeImu::ComputeJacobians() {
    /* 获取Pose_i参数 */
    VecX param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi = param_0.head<3>();
    /* 获取speedBias_i参数 */
    VecX param_1 = verticies_[1]->Parameters();
    Vec3 Vi = param_1.head<3>();
    Vec3 Bai = param_1.segment(3, 3);
    Vec3 Bgi = param_1.tail<3>();
    /* 获取Pose_j参数 */
    VecX param_2 = verticies_[2]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj = param_2.head<3>();
    /* 获取speedBias_j参数 */
    VecX param_3 = verticies_[3]->Parameters();
    Vec3 Vj = param_3.head<3>();
    Vec3 Baj = param_3.segment(3, 3);
    Vec3 Bgj = param_3.tail<3>();
    /* 获取当前IMU边界相对于偏置的雅可比 */
    double sum_dt = pre_integration_->sum_dt;
    Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);
    Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);
    Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);
    /* 检查当前IMU积分值关于上一积分时刻的雅可比的合理性 */
    if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8) {
        //ROS_WARN("numerical unstable in preintegration");
        std::cerr << "Numerical Unstable In Preintegration" << std::endl;
    }
    /* 计算当前边关于Pose_i的雅可比 */
    //if (jacobians[0])
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i; 
        jacobian_pose_i.setZero();
        /* 位置残差相对于位置&旋转的雅可比 */
        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
        /* 旋转残差相对于旋转的雅可比 */
#if 0
        /* TODO：此处0是什么意思 */
        jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
        jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif
        /* 速度残差相对于旋转的雅可比 */
        jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
        //jacobian_pose_i = sqrt_info * jacobian_pose_i;
        /* 检查当前计算得到的Jacobian是否合理 */
        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
            //ROS_WARN("numerical unstable in preintegration");
            std::cerr << "Jacobian Pose i: Numerical Unstable In Preintegration" << std::endl;
        }
        jacobians_[0] = jacobian_pose_i;
    }
    /* 计算当前边关于speedBias_i的雅可比 */
    //if (jacobians[1])
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
        jacobian_speedbias_i.setZero();
        /* 位置残差相对于速度&加速度偏置&陀螺仪偏置的雅可比 */
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
        /* 旋转残差相对于陀螺仪偏置的雅可比 */
#if 0
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
        //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
        //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif
        /* 速度残差相对于速度&加速度偏置&陀螺仪偏置的雅可比 */
        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
        /* 加速度偏置残差相对于加速度偏置的雅可比 */
        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
        /* 陀螺仪偏置残差相对于陀螺仪偏置的雅可比 */
        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
        /* 检查当前计算得到的Jacobian是否合理 */
        if (jacobian_speedbias_i.maxCoeff() > 1e8 || jacobian_speedbias_i.minCoeff() < -1e8) {
            //ROS_WARN("numerical unstable in preintegration");
            std::cerr << "Jacobian SpeedBias i: Numerical Unstable In Preintegration" << std::endl;
        }
        //jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        jacobians_[1] = jacobian_speedbias_i;
    }
    /* 计算当前边关于Pose_j的雅可比 */
    //if (jacobians[2])
    {
        Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
        jacobian_pose_j.setZero();
        /* 位置残差相对于位置的雅可比 */
        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
        /* 旋转残差相对于旋转的雅可比 */
#if 0
        jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
        Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
        jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif
        //jacobian_pose_j = sqrt_info * jacobian_pose_j;
        /* 检查当前计算得到的Jacobian是否合理 */
        if (jacobian_pose_j.maxCoeff() > 1e8 || jacobian_pose_j.minCoeff() < -1e8) {
            //ROS_WARN("numerical unstable in preintegration");
            std::cerr << "Jacobian Pose j: Numerical Unstable In Preintegration" << std::endl;
        }
        jacobians_[2] = jacobian_pose_j;
    }
    /* 计算当前边关于sppedBias_j的雅可比 */
    //if (jacobians[3])
    {
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
        jacobian_speedbias_j.setZero();
        /* 速度残差相对于速度的雅可比 */
        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
        /* 加速度偏置残差相对于加速度偏置的雅可比 */
        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
        /* 陀螺仪偏置残差相对于陀螺仪偏置的雅可比 */
        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
        //jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        /* 检查当前计算得到的Jacobian是否合理 */
        if (jacobian_speedbias_j.maxCoeff() > 1e8 || jacobian_speedbias_j.minCoeff() < -1e8) {
            //ROS_WARN("numerical unstable in preintegration");
            std::cerr << "Jacobian SpeedBias j: Numerical Unstable In Preintegration" << std::endl;
        }
        jacobians_[3] = jacobian_speedbias_j;
    }
}

}
}