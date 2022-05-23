#include "utility/utility.h"
#include "initial/initial_sfm.h"

GlobalSFM::GlobalSFM()
{
	this->feature_num = 0;
}

/*!
*  @brief 通过三角化方法恢复3D世界坐标
*  @param[in]	Pose0		第0帧相机世界姿态
*  @param[in]	Pose1		第1帧相机世界姿态
*  @param[in]	point0		特征点在第0帧相机的归一化相机坐标
*  @param[in]	point1		特征点在第1帧相机的归一化相机坐标
*  @param[out]	point_3d	特征点三角化得到的对应世界坐标
*/
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/*!
*  @brief 通过PnP方法得到当前帧相对于第l帧的位姿
*  @param[in/out]	R_initial	当前所求图像帧姿态初始值
*  @param[in/out]	P_initial	当前所求图像帧位置初始值
*  @param[in]		i			当前所求图像帧为滑窗中第i帧
*  @param[in]		sfm_f		滑窗中的所有路标点
*/
bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i,
								std::vector<SFMFeature> &sfm_f)
{
	/* 获取求解PnP问题所需的2D-3D匹配点 */
	std::vector<cv::Point2f> pts_2_vector;
	std::vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true) continue;
		Eigen::Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}

	/* 需要足够数量的匹配点才能鲁棒求解PnP问题 */
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10) return false;
	}

	/* 调用OpenCV接口求解PnP问题 */
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ) {
		return false;
	}
	cv::Rodrigues(rvec, r);
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;
}

/*!
*  @brief 通过三角化方法求解两帧图像匹配点的世界坐标
*  @param[in]		frame0		第0帧图像帧的ID
*  @param[in]		frame1		第1帧图像帧的ID
*  @param[in]		Pose0		第0帧相机世界姿态
*  @param[in]		Pose1		第1帧相机世界姿态
*  @param[in/out]	point_3d	特征点序列
*/
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	/* 检查数据合法性 */
	if (frame0 == frame1) {
		std::cerr << "Error: Triangulate Same Frame!" << std::endl;
		return;
	}

	/* 遍历所有特征，进行三角化 */
	for (int j = 0; j < feature_num; j++) {
		/* 若当前特征已经被三角化，则continue */
		if (sfm_f[j].state == true) continue;
		bool has_0 = false, has_1 = false;
		Eigen::Vector2d point0, point1;
		/* 对于每个特征，找到在frame0和frame1上的投影 */
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		/* 对投影点进行三角化 */
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}							  
	}
}

/*!
*  @brief 纯视觉SFM：求解滑窗中所有图像帧相对于第l帧的位姿以及三角化特征点坐标
*  @param[in]	frame_num	窗口总帧数:frame_count+1，frame_count为Esimator类中成员
*  @param[out]	q			窗口内图像帧的旋转四元数：相对于第l帧
*  @param[out]	T			窗口内图像帧的平移向量：相对于第l帧
*  @param[in]	l			窗口内定义的第l帧
*  @param[in]	relative_R	当前帧到第l帧的旋转矩阵
*  @param[in]	relative_T	当前帧到第l帧的平移向量
*  @param[in]	sfm_f		滑动窗口内的所有特征点
*  @param[out]	sfm_tracked_points	所有在SFM中三角化的特征点ID和坐标
*  @return		bool		纯视觉SFM是否求解成功
*/
bool GlobalSFM::construct(int& frame_num, std::vector<Eigen::Quaterniond>& q, std::vector<Vector3d>& T,
						int& l, const Eigen::Matrix3d& relative_R, const Eigen::Vector3d& relative_T,
						std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
	/* 获取滑窗中特征点数量 */
	feature_num = sfm_f.size();

	/* 将第l帧作为原点，初始化l帧和当前帧的位姿 */
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;

	/* 滑窗中的图像帧到第l帧的位姿变换 */
	std::vector<Eigen::Matrix3d> c_Rotation(frame_num);
	std::vector<Eigen::Vector3d> c_Translation(frame_num);
	std::vector<Eigen::Quaterniond> c_Quat(frame_num);
	std::vector<Eigen::Vector4d> c_rotation(frame_num);
	std::vector<Eigen::Vector3d> c_translation(frame_num);
	std::vector<Eigen::Matrix<double, 3, 4>> Pose(frame_num);

	/* 第l帧在l帧相机系中的位姿 */
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block(0, 0, 3, 3) = c_Rotation[l];
	Pose[l].block(0, 3, 3, 1) = c_Translation[l];

	/* 最后一帧即当前帧在当前相机系下的位姿 */
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block(0, 0, 3, 3) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block(0, 3, 3, 1) = c_Translation[frame_num - 1];

	/* 通过PnP求解l+1~frame_num-2帧的位姿，并三角化 */
	for (int i = l; i < frame_num - 1 ; i++)
	{
		/* 通过PnP计算出第i+1帧到第frame_num-2帧的位姿 */
		if (i > l)
		{
			Eigen::Matrix3d R_initial = c_Rotation[i - 1];
			Eigen::Vector3d P_initial = c_Translation[i - 1];
			/* 每帧PnP的位姿初始值都用上一关键帧的位姿 */
			if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
				return false;
			}
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block(0, 0, 3, 3) = c_Rotation[i];
			Pose[i].block(0, 3, 3, 1) = c_Translation[i];
		}
		/* 三角化第i帧与第frame_num-1帧之间的路标点 */
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	/* 三角化第i帧与第l帧之间的路标点 */
	for (int i = l + 1; i < frame_num - 1; i++) {
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	}

	/* 通过PnP求解0~l-1帧的位姿，并三角化 */
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		/* 每帧PnP的位姿初始值都使用上一关键帧位姿 */
		if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
			return false;
		}
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block(0, 0, 3, 3) = c_Rotation[i];
		Pose[i].block(0, 3, 3, 1) = c_Translation[i];
		/* 三角化第i帧与第l帧之间的路标点 */
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	
	/* 对于以上过程中没有三角化的特征点，以下过程全部进行三角化 */
	for (int j = 0; j < feature_num; j++)
	{
		/* 已经进行三角化的点，不再重复进行三角化 */
		if (sfm_f[j].state == true)	continue;
		/* 同时有多个相机机位观测到的路标，进行三角化 */
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}		
	}

	/* 对求解得到的滑窗中的相机姿态与路标点，进行全局BA */
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	for (int i = 0; i < frame_num; i++)
	{
		/* 姿态参数转换为ceres需要的数据格式 */
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i].data(), 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i].data(), 3);
		/* 单目问题具有姿态、位置、尺度七个自由度*/
		/* 1：通过固定l帧姿态固定单目问题中姿态的不确定性 */
		/* 2：通过固定l帧与当前帧的位置固定单目问题中位置与尺度的不确定性 */
		if (i == l) {
			problem.SetParameterBlockConstant(c_rotation[i].data());
		}
		if (i == l || i == frame_num - 1) {
			problem.SetParameterBlockConstant(c_translation[i].data());
		}
	}

	/* 向优化问题中加入重投影误差构造LM残差项 */
	for (int i = 0; i < feature_num; i++)
	{
		/* 对于没有三角化的特征，不计算重投影误差 */
		if (sfm_f[i].state != true) continue;
		problem.AddParameterBlock(sfm_f[i].position, 3);
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			/* 获取当前特征投影的Frame Id */
			int index = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[index].data(), 
									c_translation[index].data(), 
    								sfm_f[i].position);	 
		}
	}
	/* Ceres开始求解BA问题 */
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03) {
		std::cout << "Vision Only BA Converge" << std::endl;
	} else {
		std::cout << "Vision Only BA Not Converge " << std::endl;
		return false;
	}

	/* 获取世界系下的每个相机的旋转四元数 */
	for (int i = 0; i < frame_num; i++) {
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
	}

	/* 获取世界系下每个相机的位置 */
	for (int i = 0; i < frame_num; i++) {
		T[i] = -1 * (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
	}
	
	/* 获取优化后纯视觉SFM下三角化恢复的特征点世界坐标 */
	for (int i = 0; i < (int)sfm_f.size(); i++) {
		if (sfm_f[i].state) {
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
		}
	}

	/* 保存初始SFM的点云 */
	std::string DebugPath = "C:\\Users\\Administrator\\Desktop\\SFMCLOUD_AFTER_OPT.txt";
	Utility::SavePointCloudTXT(DebugPath, sfm_tracked_points);
	/* 正常返回 */
	return true;
}