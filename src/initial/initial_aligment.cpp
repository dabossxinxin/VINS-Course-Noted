#include "initial/initial_alignment.h"

/*!
*  @brief 系统初始化时求解IMU陀螺仪偏置
*  @detail 根据相机测量值与IMU测量值之间的关系，校正陀螺仪偏置误差
*		   偏置校正成功后，需要对每一帧重新计算IMU预积分值
*  @param[in/out]    all_image_frame   对齐时所有输入图像帧
*  @param[out]       Bgs               IMU陀螺仪偏置
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    Eigen::Vector3d delta_bg;
    A.setZero();
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) 
	{
        frame_j = next(frame_i);
        Eigen::MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    /* 使用Cholesky分解求AX=b */
    delta_bg = A.ldlt().solve(b);

    /* 更新陀螺仪偏置 */
    for (int i = 0; i <= WINDOW_SIZE; i++) 
	{
        Bgs[i] += delta_bg;
    }

    /* 陀螺仪偏置重新校正后，需要重新进行预积分 */
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++) 
	{
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

/*!
*  @brief 系统初始化时计算重力向量正切空间值
*  @param[in]    g0          计算的初始重力向量
*  @return       MatrixXd    重力向量正切空间的两个向量
*/
Eigen::MatrixXd TangentBasis(Vector3d &g0)
{
    Eigen::Vector3d b, c;
    Eigen::Vector3d a = g0.normalized();
    /* 论文中使用的方法 */
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp) 
	{
        tmp << 1, 0, 0;
    }
    /* 施密特正交化：保证b与a的正交性 */
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c; 
    return bc;
}

/*!
*  @brief 初始化时精化重力向量
*  @param[in]    all_image_frame   对齐时所有输入图像帧
*  @param[out]   g                 初始化时得到的初始重力向量
*  @param[out]   x                 初始化时得到的初始参数值
*/
void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    Eigen::Vector3d g0 = g.normalized() * G.norm();
    Eigen::Vector3d lx, ly;
    int all_frame_count = all_image_frame.size();
    /* 速度、重力正空间向量模长、尺度*/
    int n_state = all_frame_count * 3 + 2 + 1;
    /* 初始化信息矩阵以及残差矩阵 */
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    /* 反复在重力正切中进行优化，直到重力收敛 */
	/* TODO：只进行固定次数的迭代，重力能否成功收敛 */
    for(int k = 0; k < 4; ++k)
    {
        Eigen::MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;

            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        //double s = x(n_state - 1);
    }   
    g = g0;
}

/*!
*  @brief 初始化时求解速度、重力以及尺度
*  @param[in]    all_image_frame   对齐时所有输入图像帧
*  @param[out]   g                 初始化时得到的初始重力向量
*  @param[out]   x                 初始化时得到的初始参数值
*  @return       bool              是否成功求解速度&重力&尺度的标志
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    /* 速度、重力以及尺度 */
    int n_state = all_frame_count * 3 + 3 + 1;
    /* 初始化信息矩阵以及残差矩阵 */
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(n_state - 4);
    /* 检测优化后重力是否满足要求 */
    if((std::fabs)(g.norm() - G.norm()) > 1.0 || s < 0) 
	{
        return false;
    }
    /* 精化重力向量的值 */
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    /* 检测优化后尺度是否满足要求 */
    if(s < 0.0 ) {
        return false;   
    } else {
        return true;
    }
}

/*!
*  @brief 系统初始化时将IMU与相机对齐
*  @param[in/out]    all_image_frame   对齐时所有输入图像帧
*  @param[out]       Bgs               IMU陀螺仪偏置
*  @param[out]       g                 相机与IMU之间的标定结果
*  @param[out]       x                 相机与IMU之间的标定结果
*  @return  是否将IMU与相机对齐的标志
*/
bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, 
	Eigen::Vector3d* Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    /* 求解陀螺仪偏置 */
    solveGyroscopeBias(all_image_frame, Bgs);
    /* 初始化速度、重力向量和尺度 */
    if(LinearAlignment(all_image_frame, g, x)) {
        return true;
    } else { 
        return false; 
    }
}