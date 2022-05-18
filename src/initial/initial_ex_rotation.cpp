#include "initial/initial_ex_rotation.h"

/**@brief 初始化IMU-Camera外参标定类成员变量
*/
InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Eigen::Matrix3d::Identity());
    Rc_g.push_back(Eigen::Matrix3d::Identity());
    Rimu.push_back(Eigen::Matrix3d::Identity());
    ric = Eigen::Matrix3d::Identity();
}

/**@brief 标定IMU与相机之间的旋转参数
* @param[in]  corres                两帧图像之间的同名点
* @param[in]  delta_q_imu           两帧图像之间的IMU陀螺仪积分值
* @param[in]  calib_ric_result      相机与IMU之间的标定结果
* @return  是否成功标定IMU与相机之间的旋转参数
*/
bool InitialEXRotation::CalibrationExRotation(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres, 
                                            Eigen::Quaterniond delta_q_imu, 
                                            Eigen::Matrix3d &calib_ric_result)
{
    frame_count++;
    /* 不同相机之间的姿态变换 */
    Rc.push_back(solveRelativeR(corres));
    /* 与相机相同时间戳之内的IMU陀螺仪积分值 */
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    /* Rbc.inverse()*Rbkbk+1*Rbc：用于求解Rbkbk+1*Rbc=Rbc*Rckck+1的残差 */
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);
    /* 初始化信息矩阵 */
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();

    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++) {
        Eigen::Quaterniond r1(Rc[i]);
        Eigen::Quaterniond r2(Rc_g[i]);
        /* Rbkbk+1*Rbc=Rbc*Rckck+1的残差，并转化为角度单位 */
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        /* 设定旋转角度残差的阈值为5度，并按照这个要求设定每个数据的权重 */
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        /* 初始化四元数对应的矩阵 */
        Eigen::Matrix4d L, R;
        /* TODO：计算L矩阵，对应IMU的旋转 */
        double w = Quaterniond(Rc[i]).w();
        Eigen::Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;
        /* TODO：计算R矩阵，对应Camera的旋转 */
        Eigen::Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;
        /* 构造估计IMU-Camera之间旋转的信息矩阵 */
        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }
    /* SVD分解求解最小特征值对应的特征向量作为参数解 */
    Eigen::JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Eigen::Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    /* 1：类中记录的数据大于等于滑窗中的数据*/
    /* 2：并且倒数第二个特征值大于0.25 */
    /* 则：认为IMU-Camera外参旋转有效 */
    Eigen::Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25) {
        calib_ric_result = ric;
        return true;
    }
    else {
        return false;
    }
}

/**@brief 计算两帧相机之间的旋转矩阵
* @param[in]    corres      两帧图像之间的同名点
* @return       Matrix3d    两帧图像之间的旋转矩阵
*/
Matrix3d InitialEXRotation::solveRelativeR(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres)
{
    /* 本质矩阵求解时需要同名点大于等于9对 */
    if (corres.size() >= 9) {
        /* 获取归一化相机坐标系的U&V的坐标 */
        std::vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++) {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        /* 调用OpenCV构造本质矩阵 */
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        /* 分解本质矩阵获取两帧之间的旋转与平移 */
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);
        /* 去除旋转矩阵行列式为-1的情况 */
        if (determinant(R1) + 1.0 < 1e-09) {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = (std::max)(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = (std::max)(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Eigen::Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++) {
                ans_R_eigen(j, i) = ans_R_cv(i, j);
            }
        }
        return ans_R_eigen;
    }
    return Eigen::Matrix3d::Identity();
}

/**@brief 通过三角化测试两帧图像之间的旋转与平移的合理性
* @param[in]    l           左图像帧的特征点
* @param[in]    r           右图像帧的特征点
* @param[in]    R           两帧图像之间的旋转
* @param[in]    t           两帧图像之间的平移
* @return       double      三角化后的点在相机前方的比例
*/
double InitialEXRotation::testTriangulation(const std::vector<cv::Point2f> &l,
                                          const std::vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    /* 构造三角化3D点 */
    cv::Mat pointcloud;
    /* 构造参考帧的位置和姿态 */
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    /* 构造当前帧的位置和姿态 */
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    /* 调用OpenCV三角化函数三角化同名特征 */
    int front_count = 0;
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    
    for (int i = 0; i < pointcloud.cols; i++) {
        /* 统计同时在两相机前方的特征点数量 */
        double normal_factor = pointcloud.col(i).at<float>(3);
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0) {
            front_count++;
        }
    }
    /* 返回同时位于两相机前方的特征点数量比例 */
    return 1.0 * front_count / pointcloud.cols;
}

/**@brief 通过本质矩阵求解两帧图像之间的旋转与平移
* @param[in]     E       两帧图像之间的本质矩阵
* @param[out]    R1      两帧图像之间的旋转1
* @param[out]    R2      两帧图像之间的旋转2
* @param[out]    t1      两帧图像之间的平移1
* @param[out]    t2      两帧图像之间的平移2
*/
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
