#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "factor/integration_base.h"
#include "backend/problem.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

#include <pangolin/pangolin.h>

class Estimator
{
  public:
    Estimator();

	/*!
	*  @brief 设置VIO系统部分参数
	*/
    void setParameter();

	/*!
	*  @brief 实现IMU的预积分，通过中值积分得到当前PVQ作为优化初值
	*  @param[in]	t					输入IMU数据时间戳
	*  @param[in]	linear_acceleration	输入IMU数据线加速度
	*  @param[in]	angular_velocity	输入IMU数据角加速度
	*/
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

	/*!
	*  @brief 建立每个特征点的(camera_id, [x,y,z,u,v,vx,vy])的map
	*		  实现视觉与IMU之间的初始化以及基于滑窗的非线性优化的紧耦合
	*  @param[in]	image	输入图像帧特征
	*  @param[in]	header	输入图像帧时间戳
	*/
    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);

	/*!
	*  @brief 在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame
	*  @param[in]	_frame_stamp	重定位帧的时间戳
	*  @param[in]	_frame_index	重定位帧的索引
	*  @param[in]	_match_points	重定位帧匹配的3D特征点
	*  @param[out]	_relo_t			重定位帧位置
	*  @param[out]	_relo_r			重定位帧姿态
	*/
    void setReloFrame(double _frame_stamp, int _frame_index, std::vector<Eigen::Vector3d>& _match_points, 
				      Eigen::Vector3d& _relo_t, Eigen::Matrix3d& _relo_r);

	/*!
	*  @brief 清空/初始化滑动窗口中所有状态量
	*/
    void clearState();

	/*!
	*  @brief 视觉SFM获取初始的相机姿态及路标点
	*  @detail	bool	纯视觉SFM是否成功
	*/
    bool initialStructure();

	/*!
	*  @brief 视觉&IMU联合初始化
	*  @return	bool	相机与IMU是否成功对齐
	*/
    bool visualInitialAlign();

	/*!
	*  @brief 在滑窗中搜索与最新进来的（当前）帧之间具有足够视差和特征匹配的帧
	*  @param[out]	relative_R	两帧之间的旋转变换
	*  @param[out]	relative_T	两帧之间的平移变换
	*  @param[out]	l			滑窗中与当前帧最优匹配帧ID
	*  @return		bool		滑窗中是否找到当前帧最有匹配帧
	*/
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

	/*!
	*  @brief 基于滑动窗口的紧耦合的非线性优化
	*/
    void slideWindow();

	/*!
	*  @brief VIO非线性优化求解里程计
	*/
    void solveOdometry();

    void slideWindowNew();
    void slideWindowOld();

    void optimization();
    void backendOptimization();

	/*!
	*  @brief 优化滑窗中的状态量的具体实现
	*/
    void problemSolve();

	/*!
	*  @brief 滑窗中去掉滑窗中的最老帧
	*/
    void MargOldFrame();
    void MargNewFrame();

    void vector2double();
    void double2vector();

	/*!
	*  @brief 检测系统运行是否失败
	*/
    bool failureDetection();

	/*!
	*  @brief 向特征管理器中发布路标点
	*/
	void pubPointCloud();

	/*!
	*  @brief 获取滑窗中最后一帧的位姿
	*/
	void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
//////////////// OUR SOLVER ///////////////////
    MatXX Hprior_;
    VecX bprior_;
    VecX errprior_;
    MatXX Jprior_inv_;
    Eigen::Matrix2d project_sqrt_info_;
//////////////// OUR SOLVER //////////////////
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Eigen::Vector3d g;
    Eigen::MatrixXd Ap[2], backup_A;
    Eigen::VectorXd bp[2], backup_b;

	/*!< @brief IMU-Camera旋转外参 */
    Eigen::Matrix3d ric[NUM_OF_CAM];
	/*!< @brief IMU-Camera平移外参 */
    Eigen::Vector3d tic[NUM_OF_CAM];

	/*!< @brief 滑窗中所有关键帧的位置 */
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
	/*!< @brief 滑窗中所有关键帧的速度 */
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
	/*!< @brief 滑窗中所有关键帧的旋转 */
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
	/*!< @brief 滑窗中所有关键帧的加速度偏置 */
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
	/*!< @brief 滑窗中所有关键帧的陀螺仪偏置 */
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Eigen::Matrix3d back_R0, last_R, last_R0;
    Eigen::Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Eigen::Vector3d acc_0, gyr_0;

    std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    // MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;
    std::map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    // relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
