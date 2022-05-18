﻿#pragma once

#include "../utility/utility.h"
#include "../parameters.h"
#include <ceres/ceres.h>

using namespace Eigen;

/* IMU预积分的实现类 */
class IntegrationBase
{
  public:
    /* 禁用编译器生成该默认构造函数 */
    IntegrationBase() = delete;

    /**@brief IMU中值积分类构造函数
    * @param[in]  _acc_0            IMU加速度测量值
    * @param[in]  _gyr_0            IMU陀螺仪测量值
    * @param[in]  _linearized_ba    IMU加速度偏置
    * @param[in]  _linearized_bg    IMU陀螺仪偏置
    */
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}
    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    /**@brief 将IMU测量值加入该类中，并进行中值积分计算
    * @param[in]  dt   IMU时间戳间隔
    * @param[in]  acc  IMU加速度测量值
    * @param[in]  gyr  IMU陀螺仪测量值
    */
    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    /**@brief IMU的加速计偏置或陀螺仪偏置变动较大时，需要重新进行预积分
    * @param[in]  _linearized_ba    IMU中新的加速度偏置
    * @param[in]  _linearized_bg    IMU中新的陀螺仪偏置
    */
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++) {
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }
    }

    /**@brief IMU中值积分中由上一积分时刻计算当前积分时刻的积分值
    * @param[in]  _dt                       IMU中值积分当前时刻的时间间隔
    * @param[in]  _acc_0                    IMU中值积分开始时刻加速度值
    * @param[in]  _gyr_0                    IMU中值积分开始时刻陀螺仪值
    * @param[in]  _acc_1                    IMU中值积分结束时刻加速度值
    * @param[in]  _gyr_1                    IMU中值积分结束时刻陀螺仪值
    * @param[in]  delta_p                   IMU中值积分上一积分时刻位置积分值
    * @param[in]  delta_q                   IMU中值积分上一积分时刻姿态积分值
    * @param[in]  delta_v                   IMU中值积分上一积分时刻速度积分值
    * @param[in]  linearized_ba             IMU中值积分上一积分时刻加速度偏置
    * @param[in]  linearized_bg             IMU中值积分上一积分时刻陀螺仪偏置
    * @param[in]  result_delta_p            IMU中值积分当前积分时刻中位置积分值
    * @param[in]  result_delta_q            IMU中值积分当前积分时刻中姿态积分值
    * @param[in]  result_delta_v            IMU中值积分当前积分时刻中速度积分值
    * @param[in]  result_linearized_ba      IMU中值积分中加速度偏置积分值
    * @param[in]  result_linearized_bg      IMU中值积分中陀螺仪偏置积分值
    * @param[in]  update_jacobian           是否更新IMU中值积分值传播的雅可比
    */
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        /* 离散时间状态方程（中值积分表示）的标称状态：VINS-Mono详解[马朝伟]p10 */
        Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         
        /* 若需要重新计算IMU积分传播的雅可比，则进行下列过程 */
        if(update_jacobian) {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;
            /* 离散时间状态方程（中值积分表示）的误差状态状态转移矩阵：VINS-Mono详解[马朝伟]p14 */
            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            /* 离散时间状态方程（中值积分表示）的误差状态噪声影响矩阵：VINS-Mono详解[马朝伟]p14 */
            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  /* 0.5 * MatrixXd::Identity(3,3) * _dt */ V.block<3, 3>(3, 3);
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;
            // step_jacobian = F;
            // step_V = V;
            /* 更新IMU中值积分值传播的雅可比矩阵以及协方差矩阵 */
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }

    /**@brief 将IMU测量值加入该类中，并进行中值积分计算，类成员push_back的实现
    * @param[in]  dt      IMU时间戳间隔
    * @param[in]  _acc_1  IMU加速度测量值
    * @param[in]  _gyr_1  IMU陀螺仪测量值
    */
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
    }

    /**@brief i->j时刻内计算IMU预积分值的残差
    * @param[in]  Pi    i时刻传感器在世界系中的位置
    * @param[in]  Qi    i时刻传感器在世界系中的姿态
    * @param[in]  Vi    i时刻传感器在世界系中的速度
    * @param[in]  Bai   i时刻IMU加速度计的偏置
    * @param[in]  Bgi   i时刻IMU陀螺仪的偏置
    * @param[in]  Pj    j时刻传感器在世界系中的位置
    * @param[in]  Qj    j时刻传感器在世界系中的姿态
    * @param[in]  Vj    j时刻传感器在世界系中的速度
    * @param[in]  Baj   j时刻IMU加速度计的偏置
    * @param[in]  Bgj   j时刻IMU陀螺仪的偏置
    * @return  Eigen::Matrix<double, 15, 1> i->j时刻内IMU预积分量残差值
    */
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

public:
    /*!< @brief 每次预积分的时间周期长度 */
    double          dt;
    /*!< @brief t时刻对应的IMU测量值 */
    Eigen::Vector3d acc_0, gyr_0;
    /*!< @brief t+1时刻对应的IMU测量值 */
    Eigen::Vector3d acc_1, gyr_1;
    
    /*!< @brief k帧图像时刻对应的IMU测量值 */
    const Eigen::Vector3d linearized_acc, linearized_gyr;
    /*!< @brief 加速度计与陀螺仪零偏：k~k+1区间视为不变 */
    Eigen::Vector3d linearized_ba, linearized_bg;

    /*!< @brief 预积分误差的雅可比矩阵和对应的协方差矩阵 */
    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    /*!< @brief IMU系统噪声矩阵 */
    Eigen::Matrix<double, 18, 18> noise;

    /*!< @brief 所有IMU预积分区间的总时长 */
    double          sum_dt;
    /*!< @brief IMU位置预积分 */
    Eigen::Vector3d delta_p;
    /*!< @brief IMU旋转四元数预积分 */
    Eigen::Quaterniond delta_q;
    /*!< @brief IMU速度预积分 */
    Eigen::Vector3d delta_v;

    /*!< @brief 用于存储每次预积分时间dt的寄存器 */
    std::vector<double>             dt_buf;
    /*!< @brief 用于存储每次预积分加速度量测的寄存器 */
    std::vector<Eigen::Vector3d>    acc_buf;
    /*!< @brief 用于存储每次预积分角速度量测的寄存器 */
    std::vector<Eigen::Vector3d>    gyr_buf;

public:
    /**@brief IMU欧拉积分中由上一积分时刻计算当前积分时刻的积分值
    * @param[in]  _dt                       IMU欧拉积分当前时刻的时间间隔
    * @param[in]  _acc_0                    IMU欧拉积分开始时刻加速度值
    * @param[in]  _gyr_0                    IMU欧拉积分开始时刻陀螺仪值
    * @param[in]  _acc_1                    IMU欧拉积分结束时刻加速度值
    * @param[in]  _gyr_1                    IMU欧拉积分结束时刻陀螺仪值
    * @param[in]  delta_p                   IMU欧拉积分上一积分时刻位置积分值
    * @param[in]  delta_q                   IMU欧拉积分上一积分时刻姿态积分值
    * @param[in]  delta_v                   IMU欧拉积分上一积分时刻速度积分值
    * @param[in]  linearized_ba             IMU欧拉积分上一积分时刻加速度偏置
    * @param[in]  linearized_bg             IMU欧拉积分上一积分时刻陀螺仪偏置
    * @param[in]  result_delta_p            IMU欧拉积分当前积分时刻中位置积分值
    * @param[in]  result_delta_q            IMU欧拉积分当前积分时刻中姿态积分值
    * @param[in]  result_delta_v            IMU欧拉积分当前积分时刻中速度积分值
    * @param[in]  result_linearized_ba      IMU欧拉积分中加速度偏置积分值
    * @param[in]  result_linearized_bg      IMU欧拉积分中陀螺仪偏置积分值
    * @param[in]  update_jacobian           是否更新IMU中值积分值传播的雅可比
    */
    void eulerIntegration(double _dt, 
                          const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                          const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                          const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                          const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                          Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                          Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        /* TODO：推导一下这个式子 */
        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }     

    /**@brief 通过手动添加扰动检查雅可比矩阵的解析形式公式是否正确
    * @param[in]  _dt                 IMU欧拉积分当前时刻的时间间隔
    * @param[in]  _acc_0              IMU欧拉积分开始时刻加速度值
    * @param[in]  _gyr_0              IMU欧拉积分开始时刻陀螺仪值
    * @param[in]  _acc_1              IMU欧拉积分结束时刻加速度值
    * @param[in]  _gyr_1              IMU欧拉积分结束时刻陀螺仪值
    * @param[in]  delta_p             IMU欧拉积分上一积分时刻位置积分值
    * @param[in]  delta_q             IMU欧拉积分上一积分时刻姿态积分值
    * @param[in]  delta_v             IMU欧拉积分上一积分时刻速度积分值
    * @param[in]  linearized_ba       IMU欧拉积分上一积分时刻加速度偏置
    * @param[in]  linearized_bg       IMU欧拉积分上一积分时刻陀螺仪偏置
    */
    void checkJacobian(double _dt, 
                       const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0, 
                       const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                       const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                       const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        std::cout << "turb p       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << std::endl;
		std::cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb q       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb v       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb ba       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb bg       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb acc_0       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb _gyr_0       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb acc_1       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << std::endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
		std::cout << "turb _gyr_1       " << std::endl;
		std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
		std::cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << std::endl;
		std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
		std::cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << std::endl;
		std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
		std::cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << std::endl;
		std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
		std::cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << std::endl;
		std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
		std::cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << std::endl;
    }
};