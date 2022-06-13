#include "estimator.h"

#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_reprojection.h"
#include "backend/edge_imu.h"

#include <ostream>
#include <fstream>

using namespace myslam;

Estimator::Estimator() : f_manager{ Rs }
{
	for (size_t i = 0; i < WINDOW_SIZE + 1; i++) {
		pre_integrations[i] = nullptr;
	}
	for (auto &it : all_image_frame) {
		it.second.pre_integration = nullptr;
	}
	tmp_pre_integration = nullptr;
	clearState();
}

/*
*  @brief 设置VIO里程计所需的参数
*/
void Estimator::setParameter()
{
	/* 配置Camera与IMU之间的外参 */
	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = TIC[i];
		ric[i] = RIC[i];
	}
	//g << 0, 0, 9.81;
	std::cout << "1 Estimator::setParameter FOCAL_LENGTH: "
		<< FOCAL_LENGTH << std::endl;
	f_manager.setRic(ric);
	project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
	/* IMU时间戳与Camera时间戳之间的Offset */
	td = TD;
}

/*
*  @brief 清空VIO里程计的所有相关成员变量
*/
void Estimator::clearState()
{
	/* 清空滑窗中所有IMU与Camera的相关量 */
	for (int i = 0; i < WINDOW_SIZE + 1; i++) {
		Rs[i].setIdentity();
		Ps[i].setZero();
		Vs[i].setZero();
		Bas[i].setZero();
		Bgs[i].setZero();
		dt_buf[i].clear();
		linear_acceleration_buf[i].clear();
		angular_velocity_buf[i].clear();

		if (pre_integrations[i] != nullptr)
			delete pre_integrations[i];
		pre_integrations[i] = nullptr;
	}
	/* 清空IMU与Camera之间的外参 */
	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = Vector3d::Zero();
		ric[i] = Matrix3d::Identity();
	}
	/* 清空图像帧中IMU预积分值 */
	for (auto &it : all_image_frame) {
		if (it.second.pre_integration != nullptr) {
			delete it.second.pre_integration;
			it.second.pre_integration = nullptr;
		}
	}

	solver_flag = INITIAL;
	first_imu = false,
		sum_of_back = 0;
	sum_of_front = 0;
	frame_count = 0;
	solver_flag = INITIAL;
	initial_timestamp = 0;
	all_image_frame.clear();
	td = TD;

	if (tmp_pre_integration != nullptr) {
		delete tmp_pre_integration;
	}
	tmp_pre_integration = nullptr;

	last_marginalization_parameter_blocks.clear();

	f_manager.clearState();

	failure_occur = 0;
	relocalization_info = 0;

	drift_correct_r = Matrix3d::Identity();
	drift_correct_t = Vector3d::Zero();
}

/*
* @brief 对输入系统的每一个IMU值进行预积分
* @param[in]  dt                    两IMU测量值之间的时间差
* @param[in]  linear_acceleration   IMU量测的加速度
* @param[in]  angular_velocity      IMU量测的角速度
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
	if (!first_imu)
	{
		first_imu = true;
		acc_0 = linear_acceleration;
		gyr_0 = angular_velocity;
	}

	if (!pre_integrations[frame_count])
	{
		pre_integrations[frame_count] = new IntegrationBase{ acc_0, gyr_0, Bas[frame_count], Bgs[frame_count] };
	}

	if (frame_count != 0)
	{
		pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
		//if(solver_flag != NON_LINEAR)
		tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
		/* 将时间差&加速度&角速度数据加入成员变量 */
		dt_buf[frame_count].push_back(dt);
		linear_acceleration_buf[frame_count].push_back(linear_acceleration);
		angular_velocity_buf[frame_count].push_back(angular_velocity);

		int j = frame_count;
		/* 世界系中去除加速度偏置的加速度测量值 */
		Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
		Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
		Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
		Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
		Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
		Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
		Vs[j] += dt * un_acc;
	}
	acc_0 = linear_acceleration;
	gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
	std::cout << "Adding feature points: " << image.size() << std::endl;
	/* 添加之前检测到的特征点到feature容器中，计算每个特征的被跟踪次数以及视差 */
	/* 并且根据检测两帧之间的视差决定是否将当前帧作为关键帧 */
	if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
		marginalization_flag = MARGIN_OLD;
	}
	else {
		marginalization_flag = MARGIN_SECOND_NEW;
	}

	//ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
	//ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
	//ROS_DEBUG("Solving %d", frame_count);
	// cout << "number of feature: " << f_manager.getFeatureCount()<<endl;
	Headers[frame_count] = header;

	/* 将图像数据、时间戳、临时预积分值保存在图像帧中 */
	ImageFrame imageframe(image, header);
	imageframe.pre_integration = tmp_pre_integration;
	all_image_frame.insert(std::make_pair(header, imageframe));
	/* 更新临时预积分值 */
	tmp_pre_integration = new IntegrationBase{ acc_0, gyr_0, Bas[frame_count], Bgs[frame_count] };

	/* 1：ESTIMATE_EXTRINSIC == 0，配置文件中给出精确外参，程序中不需要标定，也不需要对外参进行优化 */
	/* 2：ESTIMATE_EXTRINSIC == 1，配置文件中给出粗略外参，程序中不需要标定，但后端需要对外参进行优化 */
	/* 3：ESTIMATE_EXTRINSIC == 2，配置文件中不给外参，需要在程序中对外参进行标定，也需要后端进行优化 */
	if (ESTIMATE_EXTRINSIC == 2)
	{
		std::cout << "Calibrating Extrinsic Param, Rotation Movement Is Needed" << std::endl;
		if (frame_count != 0)
		{
			std::vector<pair<Eigen::Vector3d, Eigen::Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
			Eigen::Matrix3d calib_ric;
			if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
			{
				ric[0] = calib_ric;
				RIC[0] = calib_ric;
				ESTIMATE_EXTRINSIC = 1;
			}
		}
	}

	if (solver_flag == INITIAL)
	{
		if (frame_count == WINDOW_SIZE)
		{
			/* 执行Visual-IMU联合初始化 */
			bool result = false;
			if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
				result = initialStructure();
				initial_timestamp = header;
			}
			/* 联合初始化成功则进行一次非线性优化 */
			if (result) {
				solver_flag = NON_LINEAR;
				solveOdometry();
				slideWindow();
				f_manager.removeFailures();
				std::cout << "Initialization Finish!" << std::endl;
				/* 记录滑窗中最后一帧的位姿 */
				last_R = Rs[WINDOW_SIZE];
				last_P = Ps[WINDOW_SIZE];
				/* 记录滑窗中第一帧的位姿 */
				last_R0 = Rs[0];
				last_P0 = Ps[0];
				/* 联合初始化不成功则进行一次滑窗操作 */
			}
			else {
				slideWindow();
			}
		}
		else {
			frame_count++;
		}
	}
	else
	{
		TicToc t_solve;
		/* 执行非线性优化 */
		solveOdometry();
		/* 检测系统运行是否失败，若失败则重置估计器 */
		if (failureDetection())
		{
			failure_occur = 1;
			clearState();
			setParameter();
			return;
		}

		TicToc t_margin;
		slideWindow();
		f_manager.removeFailures();
		//ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
		// prepare output of VINS
		key_poses.clear();
		for (int i = 0; i <= WINDOW_SIZE; i++)
			key_poses.push_back(Ps[i]);

		last_R = Rs[WINDOW_SIZE];
		last_P = Ps[WINDOW_SIZE];
		last_R0 = Rs[0];
		last_P0 = Ps[0];
	}
}

bool Estimator::initialStructure()
{
	TicToc t_sfm;
	/* 检测IMU的加速度激励是否足够 */
	{
		std::map<double, ImageFrame>::iterator frame_it;
		Vector3d sum_g;
		for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
		{
			double dt = frame_it->second.pre_integration->sum_dt;
			Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
			sum_g += tmp_g;
		}
		Vector3d aver_g;
		aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
		double var = 0;
		for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
		{
			double dt = frame_it->second.pre_integration->sum_dt;
			Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
			var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
			//cout << "frame g " << tmp_g.transpose() << endl;
		}
		var = sqrt(var / ((int)all_image_frame.size() - 1));
		//ROS_WARN("IMU variation %f!", var);
		if (var < 0.25)
		{
			// ROS_INFO("IMU excitation not enouth!");
			//return false;
		}
	}

	/* 开始进行纯视觉SFM */
	/* 获取SFM中输入的所有特征点信息 */
	std::vector<SFMFeature> sfm_f;
	for (auto &it_per_id : f_manager.feature)
	{
		int imu_j = it_per_id.start_frame - 1;
		SFMFeature tmp_feature;
		tmp_feature.state = false;
		tmp_feature.id = it_per_id.feature_id;
		for (auto &it_per_frame : it_per_id.feature_per_frame)
		{
			imu_j++;
			Eigen::Vector3d pts_j = it_per_frame.point;
			tmp_feature.observation.push_back(std::make_pair(imu_j, Eigen::Vector2d{ pts_j.x(), pts_j.y() }));
		}
		sfm_f.push_back(tmp_feature);
	}

	/* 获取SFM中的参考帧索引l */
	int l = 0;
	Eigen::Matrix3d relative_R;
	Eigen::Vector3d relative_T;
	if (!relativePose(relative_R, relative_T, l))
	{
		std::cout << "Not Enough Features or Parallax; Move Device Around" << std::endl;
		return false;
	}

	/* 视觉SFM恢复滑窗中所有帧的位姿和路标点 */
	GlobalSFM sfm;
	int count_new = frame_count + 1;
	std::map<int, Eigen::Vector3d> sfm_tracked_points;
	std::vector<Eigen::Quaterniond> Q(frame_count + 1);
	std::vector<Eigen::Vector3d> T(frame_count + 1);
	if (!sfm.construct(count_new, Q, T, l,
		relative_R, relative_T,
		sfm_f, sfm_tracked_points))
	{
		std::cout << "Global SFM Failed!" << std::endl;
		marginalization_flag = MARGIN_OLD;
		return false;
	}

	/* 通过PnP恢复所有帧的位置和姿态 */
	std::map<double, ImageFrame>::iterator frame_it;
	std::map<int, Vector3d>::iterator it;
	frame_it = all_image_frame.begin();
	for (int i = 0; frame_it != all_image_frame.end(); ++frame_it)
	{
		cv::Mat r, rvec, t, D, tmp_r;
		if ((frame_it->first) == Headers[i])
		{
			frame_it->second.is_key_frame = true;
			frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
			frame_it->second.T = T[i];
			i++;
			continue;
		}
		if ((frame_it->first) > Headers[i])
		{
			i++;
		}
		Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
		Vector3d P_inital = -R_inital * T[i];
		cv::eigen2cv(R_inital, tmp_r);
		cv::Rodrigues(tmp_r, rvec);
		cv::eigen2cv(P_inital, t);

		frame_it->second.is_key_frame = false;
		vector<cv::Point3f> pts_3_vector;
		vector<cv::Point2f> pts_2_vector;
		for (auto &id_pts : frame_it->second.points)
		{
			int feature_id = id_pts.first;
			for (auto &i_p : id_pts.second)
			{
				it = sfm_tracked_points.find(feature_id);
				if (it != sfm_tracked_points.end())
				{
					Vector3d world_pts = it->second;
					cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
					pts_3_vector.push_back(pts_3);
					Vector2d img_pts = i_p.second.head<2>();
					cv::Point2f pts_2(img_pts(0), img_pts(1));
					pts_2_vector.push_back(pts_2);
				}
			}
		}
		cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		if (pts_3_vector.size() < 6)
		{
			std::cout << "Not Enough Points For Solve PnP Pts_3_vector Size " << pts_3_vector.size() << std::endl;
			return false;
		}
		if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
		{
			std::cout << " Solve PnP Fail!" << std::endl;
			return false;
		}
		cv::Rodrigues(rvec, r);
		MatrixXd R_pnp, tmp_R_pnp;
		cv::cv2eigen(r, tmp_R_pnp);
		R_pnp = tmp_R_pnp.transpose();
		MatrixXd T_pnp;
		cv::cv2eigen(t, T_pnp);
		T_pnp = R_pnp * (-T_pnp);
		frame_it->second.R = R_pnp * RIC[0].transpose();
		frame_it->second.T = T_pnp;
	}

	/* 将IMU预积分结果与视觉SFM结果对齐 */
	if (visualInitialAlign())
	{
		return true;
	}
	else
	{
		std::cout << "MisAlign Visual Structure With IMU" << std::endl;
		return false;
	}
}

/*!
*  @brief 视觉&IMU联合初始化
*  @detail 1：计算陀螺仪偏置，尺度，重力加速度和速度
*		   2：重新计算特征点深度、预积分值
*		   3：将位置、速度、深度按尺度进行缩放
*		   4：根据重力方向，更新所有图像帧在惯性系下的位置、姿态和速度
*  @return	bool	相机与IMU是否成功对齐
*/
bool Estimator::visualInitialAlign()
{
	TicToc t_g;
	/* 相机与IMU进行对齐，获取重力，速度，尺度与陀螺仪偏置信息 */
	Eigen::VectorXd x;
	bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
	if (!result)
	{
		std::cerr << "Visual Inertial Align Solve G Failed..." << std::endl;
		return false;
	}

	/* 传递所有图像帧的位姿，并将其置为关键帧 */
	for (int i = 0; i <= frame_count; i++)
	{
		/* 1：此处旋转矩阵添加了IMU-Camera外参的影响 */
		/* 2：平移向量并没有添加外参影响，后面会重新计算一个平移 */
		Eigen::Matrix3d Ri = all_image_frame[Headers[i]].R;
		Eigen::Vector3d Pi = all_image_frame[Headers[i]].T;
		Ps[i] = Pi;
		Rs[i] = Ri;
		all_image_frame[Headers[i]].is_key_frame = true;
	}

	/* 重新计算所有特征点的深度：先将特征点深度设为-1 */
	Eigen::VectorXd dep = f_manager.getDepthVector();
	for (int i = 0; i < dep.size(); i++)
	{
		dep[i] = -1;
	}
	f_manager.clearDepth(dep);

	/* 重新计算所有特征点的深度：三角化计算特征深度 */
	Eigen::Vector3d TIC_TMP[NUM_OF_CAM];
	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		/* 由于Ps此时依然以第L帧相机系为参考，所以此时tic设置为0 */
		TIC_TMP[i].setZero();
	}
	ric[0] = RIC[0];
	f_manager.setRic(ric);
	f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

	/* 陀螺仪偏置发生了改变，需要重新进行预积分 */
	double s = (x.tail<1>())(0);
	for (int i = 0; i <= WINDOW_SIZE; i++)
	{
		/* 在更新了陀螺仪偏置误差后，也重新进行了预积分，然后当时预积分 */
		/* 发生在all_img_frame中，此时会发生在滑窗中的帧中 */
		pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
	}

	/* 更新Ps：这里的Ps依然以第l帧相机系作为参考，表示IMU b0系到bk系的位移向量 */
	for (int i = frame_count; i >= 0; i--)
	{
		Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
	}
	/* 更新Vs：这里的Vs依然以第l帧相机系作为参考，表示IMU bk帧的速度 */
	int kv = -1;
	std::map<double, ImageFrame>::iterator frame_i;
	for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
	{
		if (frame_i->second.is_key_frame)
		{
			/* 速度求解出来本身在Body系下，尺度为真实尺度，因此不需要使用s更新 */
			kv++;
			Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
		}
	}
	/* 更新深度 */
	for (auto &it_per_id : f_manager.feature)
	{
		/* 1：被观测次数小于2此的特征，不计算其深度 */
		/* 2：滑窗中首次观测到该特征的帧序号较为靠后，不计算其深度 */
		it_per_id.used_num = it_per_id.feature_per_frame.size();
		if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
		{
			continue;
		}
		it_per_id.estimated_depth *= s;
	}

	/* 根据重力方向，更新P、V、Q在惯性系下的位置 */
	Eigen::Matrix3d R0 = Utility::g2R(g);// Rw->cl
	/* 令第一帧IMU坐标系在世界系下的yaw为0 */
	double yaw = Utility::R2ypr(R0 * Rs[0]).x();
	R0 = Utility::ypr2R(Eigen::Vector3d{ -yaw, 0, 0 }) * R0;
	g = R0 * g;
	Eigen::Matrix3d rot_diff = R0;
	for (int i = 0; i <= frame_count; i++)
	{
		Ps[i] = rot_diff * Ps[i];
		Rs[i] = rot_diff * Rs[i];
		Vs[i] = rot_diff * Vs[i];
	}
	return true;
}

/*!
*  @brief 在滑窗中搜索与最新进来的（当前）帧之间具有足够视差和特征匹配的帧
*  @param[out]	relative_R	两帧之间的旋转变换
*  @param[out]	relative_T	两帧之间的平移变换
*  @param[out]	l			滑窗中与当前帧最优匹配帧ID
*  @return		bool		滑窗中是否找到当前帧最有匹配帧
*/
bool Estimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
	for (int i = 0; i < WINDOW_SIZE; i++)
	{
		/* 获取第i帧与当前帧的匹配特征点 */
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
		corres = f_manager.getCorresponding(i, WINDOW_SIZE);
		if (corres.size() > 20)
		{
			/* 计算所有匹配点的视差 */
			double sum_parallax = 0;
			double average_parallax;
			for (int j = 0; j < int(corres.size()); j++)
			{
				Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
				Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
				double parallax = (pts_0 - pts_1).norm();
				sum_parallax = sum_parallax + parallax;
			}
			average_parallax = 1.0 * sum_parallax / int(corres.size());
			/* 在像素坐标系中平均视差大于30个像素 */
			if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
			{
				l = i;
				return true;
			}
		}
	}
	return false;
}

/*!
*  @brief VIO非线性优化求解里程计
*  @detail 对滑窗中的状态量进行非线性优化
*/
void Estimator::solveOdometry()
{
	/* 滑窗中没有足够的关键帧，不进行优化 */
	if (frame_count < WINDOW_SIZE) {
		return;
	}
	if (solver_flag == NON_LINEAR)
	{
		TicToc t_tri;
		/* 滑窗中进来新的帧后，三角化恢复一些路标点，增加视觉约束信息 */
		f_manager.triangulate(Ps, tic, ric);
		//cout << "triangulation costs : " << t_tri.toc() << endl;        
		backendOptimization();
	}
}

void Estimator::vector2double()
{
	for (int i = 0; i <= WINDOW_SIZE; i++)
	{
		para_Pose[i][0] = Ps[i].x();
		para_Pose[i][1] = Ps[i].y();
		para_Pose[i][2] = Ps[i].z();
		Quaterniond q{ Rs[i] };
		para_Pose[i][3] = q.x();
		para_Pose[i][4] = q.y();
		para_Pose[i][5] = q.z();
		para_Pose[i][6] = q.w();

		para_SpeedBias[i][0] = Vs[i].x();
		para_SpeedBias[i][1] = Vs[i].y();
		para_SpeedBias[i][2] = Vs[i].z();

		para_SpeedBias[i][3] = Bas[i].x();
		para_SpeedBias[i][4] = Bas[i].y();
		para_SpeedBias[i][5] = Bas[i].z();

		para_SpeedBias[i][6] = Bgs[i].x();
		para_SpeedBias[i][7] = Bgs[i].y();
		para_SpeedBias[i][8] = Bgs[i].z();
	}
	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		para_Ex_Pose[i][0] = tic[i].x();
		para_Ex_Pose[i][1] = tic[i].y();
		para_Ex_Pose[i][2] = tic[i].z();
		Eigen::Quaterniond q{ ric[i] };
		para_Ex_Pose[i][3] = q.x();
		para_Ex_Pose[i][4] = q.y();
		para_Ex_Pose[i][5] = q.z();
		para_Ex_Pose[i][6] = q.w();
	}

	Eigen::VectorXd dep = f_manager.getDepthVector();
	for (int i = 0; i < f_manager.getFeatureCount(); i++)
	{
		para_Feature[i][0] = dep(i);
	}
	if (ESTIMATE_TD)
	{
		para_Td[0][0] = td;
	}
}

void Estimator::double2vector()
{
	Eigen::Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
	Eigen::Vector3d origin_P0 = Ps[0];

	if (failure_occur)
	{
		origin_R0 = Utility::R2ypr(last_R0);
		origin_P0 = last_P0;
		failure_occur = 0;
	}
	Eigen::Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
		para_Pose[0][3],
		para_Pose[0][4],
		para_Pose[0][5])
		.toRotationMatrix());
	double y_diff = origin_R0.x() - origin_R00.x();
	//TODO
	Eigen::Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
	if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
	{
		//ROS_DEBUG("euler singular point!");
		rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
			para_Pose[0][3],
			para_Pose[0][4],
			para_Pose[0][5])
			.toRotationMatrix()
			.transpose();
	}

	for (int i = 0; i <= WINDOW_SIZE; i++)
	{
		Rs[i] = rot_diff * Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

		Ps[i] = rot_diff * Eigen::Vector3d(para_Pose[i][0] - para_Pose[0][0],
			para_Pose[i][1] - para_Pose[0][1],
			para_Pose[i][2] - para_Pose[0][2]) +
			origin_P0;

		Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
			para_SpeedBias[i][1],
			para_SpeedBias[i][2]);

		Bas[i] = Vector3d(para_SpeedBias[i][3],
			para_SpeedBias[i][4],
			para_SpeedBias[i][5]);

		Bgs[i] = Vector3d(para_SpeedBias[i][6],
			para_SpeedBias[i][7],
			para_SpeedBias[i][8]);
	}

	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		tic[i] = Vector3d(para_Ex_Pose[i][0],
			para_Ex_Pose[i][1],
			para_Ex_Pose[i][2]);
		ric[i] = Quaterniond(para_Ex_Pose[i][6],
			para_Ex_Pose[i][3],
			para_Ex_Pose[i][4],
			para_Ex_Pose[i][5])
			.toRotationMatrix();
	}

	Eigen::VectorXd dep = f_manager.getDepthVector();
	for (int i = 0; i < f_manager.getFeatureCount(); i++)
	{
		dep(i) = para_Feature[i][0];
	}
	f_manager.setDepth(dep);
	this->pubPointCloud();

	if (ESTIMATE_TD)
	{
		td = para_Td[0][0];
	}

	// relative info between two loop frame
	if (relocalization_info)
	{
		Matrix3d relo_r;
		Vector3d relo_t;
		relo_r = rot_diff * Eigen::Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
		relo_t = rot_diff * Eigen::Vector3d(relo_Pose[0] - para_Pose[0][0],
			relo_Pose[1] - para_Pose[0][1],
			relo_Pose[2] - para_Pose[0][2]) +
			origin_P0;
		double drift_correct_yaw;
		drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
		drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
		drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
		relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
		relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
		relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
		//cout << "vins relo " << endl;
		//cout << "vins relative_t " << relo_relative_t.transpose() << endl;
		//cout << "vins relative_yaw " <<relo_relative_yaw << endl;
		relocalization_info = 0;
	}
}

bool Estimator::failureDetection()
{
	if (f_manager.last_track_num < 2)
	{
		std::cout << "Little Feature Track: " << f_manager.last_track_num << std::endl;
		return true;
	}
	if (Bas[WINDOW_SIZE].norm() > 2.5)
	{
		std::cout << "Big IMU acc Bias Estimation: " << Bas[WINDOW_SIZE].norm() << std::endl;
		return true;
	}
	if (Bgs[WINDOW_SIZE].norm() > 1.0)
	{
		std::cout << "Big IMU gyr Bias Estimate: " << Bgs[WINDOW_SIZE].norm() << std::endl;
		return true;
	}
	/*
	if (tic(0) > 1)
	{
		//ROS_INFO(" big extri param estimation %d", tic(0) > 1);
		return true;
	}
	*/
	Vector3d tmp_P = Ps[WINDOW_SIZE];
	if ((tmp_P - last_P).norm() > 5)
	{
		/*std::cout << "Big Translation" << std::endl;
		return true;*/
	}
	if (abs(tmp_P.z() - last_P.z()) > 1)
	{
		/*std::cout << "Big Z Translation" << std::endl;
		return true;*/
	}
	Matrix3d tmp_R = Rs[WINDOW_SIZE];
	Matrix3d delta_R = tmp_R.transpose() * last_R;
	Quaterniond delta_Q(delta_R);
	double delta_angle;
	delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
	if (delta_angle > 50)
	{
		//ROS_INFO(" big delta_angle ");
		//return true;
	}
	return false;
}

/*!
*  @brief 向特征管理器中发布路标点
*/
void Estimator::pubPointCloud()
{
	for (auto &it_per_id : this->f_manager.feature)
	{
		int used_num;
		used_num = it_per_id.feature_per_frame.size();
		if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
			continue;
		if (it_per_id.solve_flag != 1)
			continue;

		int imu_i = it_per_id.start_frame;
		Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point*it_per_id.estimated_depth;
		Eigen::Vector3d w_pts_i = this->Rs[imu_i] * (this->ric[0] * pts_i + this->tic[0]) + this->Ps[imu_i];

		it_per_id.world_pts = w_pts_i;
	}
}

/*!
*  @brief 获取滑窗中最后一帧的位姿
*/
void Estimator::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
	Eigen::Matrix3d Rwc;
	Eigen::Vector3d twc;
	Rwc = Rs[WINDOW_SIZE] * ric[0];
	twc = Ps[WINDOW_SIZE] + Rs[WINDOW_SIZE] * tic[0];

	M.m[0] = Rwc(0, 0);
	M.m[1] = Rwc(1, 0);
	M.m[2] = Rwc(2, 0);
	M.m[3] = 0.0;

	M.m[4] = Rwc(0, 1);
	M.m[5] = Rwc(1, 1);
	M.m[6] = Rwc(2, 1);
	M.m[7] = 0.0;

	M.m[8] = Rwc(0, 2);
	M.m[9] = Rwc(1, 2);
	M.m[10] = Rwc(2, 2);
	M.m[11] = 0.0;

	M.m[12] = twc(0);
	M.m[13] = twc(1);
	M.m[14] = twc(2);
	M.m[15] = 1.0;
}

/*!
*  @brief 滑窗中去掉滑窗中的最老帧
*  @detail 1：构造优化问题，并且将滑窗中所有帧的状态加入优化问题
*          2：将最老帧对应的IMU测量以及视觉测量加入优化问题中
*          3：在优化问题中构造Hessian矩阵，并加上上一次边缘化形成的先验Hessian
*		   4：对于合成的新Hessian矩阵，边缘化掉最老帧所对应的矩阵块
*/
void Estimator::MargOldFrame()
{
	backend::LossFunction *lossfunction;
	lossfunction = new backend::CauchyLoss(1.0);

	/* 构建优化问题 */
	backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
	std::vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
	std::vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
	/* 当前优化问题中，将滑窗中的每一帧的姿态以及IMU-Camera外参姿态均认为Pose */
	int pose_dim = 0;

	/* 将IMU-Camera外参加入到优化节点中 */
	std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
	{
		Eigen::VectorXd pose(7);
		pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
		vertexExt->SetParameters(pose);
		problem.AddVertex(vertexExt);
		pose_dim += vertexExt->LocalDimension();
	}

	/* 将相机位姿，滑窗中每一帧的速度以及陀螺仪偏置、加速度计偏置加入到优化节点中 */
	for (int i = 0; i < WINDOW_SIZE + 1; i++)
	{
		std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
		Eigen::VectorXd pose(7);
		pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
		vertexCam->SetParameters(pose);
		vertexCams_vec.push_back(vertexCam);
		problem.AddVertex(vertexCam);
		pose_dim += vertexCam->LocalDimension();

		shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
		Eigen::VectorXd vb(9);
		vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
			para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
			para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
		vertexVB->SetParameters(vb);
		vertexVB_vec.push_back(vertexVB);
		problem.AddVertex(vertexVB);
		pose_dim += vertexVB->LocalDimension();
	}

	/* 将第1帧IMU的测量信息加入到欧化问题边中 */
	{
		/* 若第一帧IMU积分时间过长，其测量就不准了，此时认为这一帧没有对应的IMU测量 */
		if (pre_integrations[1]->sum_dt < 10.0)
		{
			std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[1]));
			std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
			edge_vertex.push_back(vertexCams_vec[0]);
			edge_vertex.push_back(vertexVB_vec[0]);
			edge_vertex.push_back(vertexCams_vec[1]);
			edge_vertex.push_back(vertexVB_vec[1]);
			imuEdge->SetVertex(edge_vertex);
			problem.AddEdge(imuEdge);
		}
	}

	/* 将第0帧对应的视觉残差加入到优化问题中 */
	{
		int feature_index = -1;
		/* 遍历滑窗中所有的特征，构造视觉测量残差 */
		for (auto &it_per_id : f_manager.feature)
		{
			/* 1：特征点被观测次数小于两次，认为是不鲁棒的特征，不加入视觉测量中 */
			/* 2：特征点刚被滑窗中的帧观测到，鲁棒性可能不强，不加入视觉测量中 */
			it_per_id.used_num = it_per_id.feature_per_frame.size();
			if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
				continue;

			++feature_index;

			/* 只需要将首次观测帧为第0帧的特征加入到优化中的视觉测量部分 */
			int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
			if (imu_i != 0)
				continue;

			Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

			std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
			VecX inv_d(1);
			inv_d << para_Feature[feature_index][0];
			verterxPoint->SetParameters(inv_d);
			problem.AddVertex(verterxPoint);

			/* 满足上述条件的特征，构造视觉残差边加入到优化问题中 */
			for (auto &it_per_frame : it_per_id.feature_per_frame)
			{
				imu_j++;
				if (imu_i == imu_j)
					continue;

				Vector3d pts_j = it_per_frame.point;

				std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
				std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
				edge_vertex.push_back(verterxPoint);
				edge_vertex.push_back(vertexCams_vec[imu_i]);
				edge_vertex.push_back(vertexCams_vec[imu_j]);
				edge_vertex.push_back(vertexExt);

				/* 视觉残差能较好满足正态分布，加入柯西鲁棒核函数 */
				edge->SetVertex(edge_vertex);
				edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);
				edge->SetLossFunction(lossfunction);
				problem.AddEdge(edge);
			}
		}
	}

	/* 计算边缘化最老帧形成的先验信息 */
	{
		/* 上一次边缘化已经形成的先验信息 */
		if (Hprior_.rows() > 0)
		{
			problem.SetHessianPrior(Hprior_);
			problem.SetbPrior(bprior_);
			problem.SetErrPrior(errprior_);
			problem.SetJtPrior(Jprior_inv_);
			problem.ExtendHessiansPriorSize(15);
		}
		/* 上一次并没有进行边缘化的流程 */
		else
		{
			Hprior_ = MatXX(pose_dim, pose_dim);
			Hprior_.setZero();
			bprior_ = VecX(pose_dim);
			bprior_.setZero();
			problem.SetHessianPrior(Hprior_);
			problem.SetbPrior(bprior_);
		}
	}

	/* 计算边缘化后的先验信息，并赋值给estimator类中的成员变量 */
	std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
	marg_vertex.push_back(vertexCams_vec[0]);
	marg_vertex.push_back(vertexVB_vec[0]);
	problem.Marginalize(marg_vertex, pose_dim);
	Hprior_ = problem.GetHessianPrior();
	bprior_ = problem.GetbPrior();
	errprior_ = problem.GetErrPrior();
	Jprior_inv_ = problem.GetJtPrior();
}

/*!
*  @brief 滑窗中边缘化掉次新帧
*  @detail 1：构造优化问题，并且将滑窗中所有帧的状态加入优化问题
*          2：在优化问题中构造Hessian矩阵，并加上上一次边缘化形成的先验Hessian
*		   3：对于合成的新Hessian矩阵，边缘化掉次新帧所对应的矩阵块
*          4：由于没有在优化问题中加入边，因此优化问题中构造的Hessian为0矩阵
*/
void Estimator::MargNewFrame()
{
	/* 构建优化问题，优化节点 */
	backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
	vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
	vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
	//vector<backend::Point3d> points;
	int pose_dim = 0;

	/* 构建外参优化节点，并加入到优化问题当中 */
	shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
	{
		Eigen::VectorXd pose(7);
		pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
		vertexExt->SetParameters(pose);
		problem.AddVertex(vertexExt);
		pose_dim += vertexExt->LocalDimension();
	}

	/* 构建相机PVQ Ba Bg相关节点，并将滑窗中的所有帧对应的该节点都加入优化问题 */
	for (int i = 0; i < WINDOW_SIZE + 1; i++)
	{
		shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
		Eigen::VectorXd pose(7);
		pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
		vertexCam->SetParameters(pose);
		vertexCams_vec.push_back(vertexCam);
		problem.AddVertex(vertexCam);
		pose_dim += vertexCam->LocalDimension();

		shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
		Eigen::VectorXd vb(9);
		vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
			para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
			para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
		vertexVB->SetParameters(vb);
		vertexVB_vec.push_back(vertexVB);
		problem.AddVertex(vertexVB);
		pose_dim += vertexVB->LocalDimension();
	}

	/* 获取上一把优化的先验，并将该先验加入到当前先验的维护中 */
	{
		if (Hprior_.rows() > 0)
		{
			problem.SetHessianPrior(Hprior_);
			problem.SetbPrior(bprior_);
			problem.SetErrPrior(errprior_);
			problem.SetJtPrior(Jprior_inv_);
			problem.ExtendHessiansPriorSize(15);
		}
		else
		{
			Hprior_ = MatXX(pose_dim, pose_dim);
			Hprior_.setZero();
			bprior_ = VecX(pose_dim);
			bprior_.setZero();
		}
	}

	/* 边缘化掉次新帧 */
	std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
	marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);
	marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
	problem.Marginalize(marg_vertex, pose_dim);
	Hprior_ = problem.GetHessianPrior();
	bprior_ = problem.GetbPrior();
	errprior_ = problem.GetErrPrior();
	Jprior_inv_ = problem.GetJtPrior();
}

/*!
*  @brief 优化滑窗中的状态量的具体实现
*/
void Estimator::problemSolve()
{
	/* 使用cauchy鲁棒核函数进行抗噪 */
	backend::LossFunction *lossfunction;
	lossfunction = new backend::CauchyLoss(1.0);
	//lossfunction = new backend::TukeyLoss(1.0);

	/* step1：构建优化问题，创建待优化的节点 */
	backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
	std::vector<std::shared_ptr<backend::VertexPose>> vertexCams_vec;
	std::vector<std::shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
	int pose_dim = 0;

	/* 将IMU-Camera外参加入到优化问题中，根据ESTIMATE_EXTRINSIC调节该参数是否固定 */
	std::shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
	{
		Eigen::VectorXd pose(7);
		pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
		vertexExt->SetParameters(pose);

		if (!ESTIMATE_EXTRINSIC)
		{
			std::cout << "Fix Extrinsic Param" << std::endl;
			vertexExt->SetFixed();
		}
		else {
			std::cout << "Estimate IMU-Camera Extrinsic Param" << std::endl;
		}
		problem.AddVertex(vertexExt);
		pose_dim += vertexExt->LocalDimension();
	}

	/* 将滑动窗口中每一帧的p、q、v、ba、bg添加进优化问题中 */
	for (int i = 0; i < WINDOW_SIZE + 1; i++)
	{
		std::shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
		Eigen::VectorXd pose(7);
		pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
		vertexCam->SetParameters(pose);
		vertexCams_vec.push_back(vertexCam);
		problem.AddVertex(vertexCam);
		pose_dim += vertexCam->LocalDimension();

		std::shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
		Eigen::VectorXd vb(9);
		vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
			para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
			para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
		vertexVB->SetParameters(vb);
		vertexVB_vec.push_back(vertexVB);
		problem.AddVertex(vertexVB);
		pose_dim += vertexVB->LocalDimension();
	}

	/* 将IMU测量残差加入到优化问题中 */
	for (int i = 0; i < WINDOW_SIZE; i++)
	{
		int j = i + 1;
		/* 若IMU连续积分时间过长，认为这个测量值误差很大了，不加入到优化中 */
		if (pre_integrations[j]->sum_dt > 10.0)
			continue;

		std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[j]));
		std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
		edge_vertex.push_back(vertexCams_vec[i]);
		edge_vertex.push_back(vertexVB_vec[i]);
		edge_vertex.push_back(vertexCams_vec[j]);
		edge_vertex.push_back(vertexVB_vec[j]);
		imuEdge->SetVertex(edge_vertex);
		problem.AddEdge(imuEdge);
	}

	/* 将视觉残差加入到优化问题中 */
	std::vector<std::shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
	{
		int feature_index = -1;
		/* 遍历特征管理器中的每一个特征，将视觉重投影误差加入优化问题中 */
		for (auto &it_per_id : f_manager.feature)
		{
			/* 1：特征中被观测次数小于2次的不加入到优化问题中 */
			/* 2：特征中首次被观测的帧为滑窗中的后面几帧，不加入到优化问题中 */
			it_per_id.used_num = it_per_id.feature_per_frame.size();
			if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
			{
				continue;
			}

			++feature_index;

			int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
			Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

			/* 向优化问题中添加逆深度节点 */
			std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
			VecX inv_d(1);
			inv_d << para_Feature[feature_index][0];
			verterxPoint->SetParameters(inv_d);
			problem.AddVertex(verterxPoint);
			vertexPt_vec.push_back(verterxPoint);

			/* 遍历同一特征在所有帧中的观测 */
			for (auto &it_per_frame : it_per_id.feature_per_frame)
			{
				imu_j++;
				if (imu_i == imu_j)
					continue;

				Vector3d pts_j = it_per_frame.point;

				std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
				std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
				edge_vertex.push_back(verterxPoint);
				edge_vertex.push_back(vertexCams_vec[imu_i]);
				edge_vertex.push_back(vertexCams_vec[imu_j]);
				edge_vertex.push_back(vertexExt);

				edge->SetVertex(edge_vertex);
				edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);
				edge->SetLossFunction(lossfunction);
				problem.AddEdge(edge);
			}
		}
	}

	/* 向优化问题中添加先验信息 */
	{
		/* 如果有先验信息，则将先验信息加入到优化问题中，否则忽略 */
        if (Hprior_.rows() > 0)
        {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

            problem.SetHessianPrior(Hprior_);
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15);
        }
    }

	//problem.SetNonLinearMethod(myslam::backend::Problem::NonLinearMethod::Dog_Leg);
    problem.Solve(10);

    /* 更新先验信息矩阵 */
    if (Hprior_.rows() > 0)
    {
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << bprior_.norm() << std::endl;
        std::cout << "                     " << errprior_.norm() << std::endl;
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        std::cout << "             after: " << bprior_.norm() << std::endl;
        std::cout << "                    " << errprior_.norm() << std::endl;
    }

    /* 更新滑窗中每一帧的状态 */
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        VecX p = vertexCams_vec[i]->Parameters();
        for (int j = 0; j < 7; ++j)
        {
            para_Pose[i][j] = p[j];
        }

        VecX vb = vertexVB_vec[i]->Parameters();
        for (int j = 0; j < 9; ++j)
        {
            para_SpeedBias[i][j] = vb[j];
        }
    }

    /* 更新滑窗中特征的逆深度 */
    for (int i = 0; i < vertexPt_vec.size(); ++i)
    {
        VecX f = vertexPt_vec[i]->Parameters();
        para_Feature[i][0] = f[0];
    }

}

void Estimator::backendOptimization()
{
	TicToc t_solver;
	/* 将滑窗中的状态量向优化中待使用的状态量转移 */
	vector2double();
	// 构建求解器
	problemSolve();
	/* 优化后的状态量转移到滑窗中已定义的状态量中 */
	double2vector();
	//ROS_INFO("whole time for solver: %f", t_solver.toc());

	/* 维护滑窗的边缘化信息 */
	TicToc t_whole_marginalization;
	if (marginalization_flag == MARGIN_OLD)
	{
		vector2double();
		MargOldFrame();

		std::unordered_map<long, double*> addr_shift; // prior 中对应的保留下来的参数地址
		for (int i = 1; i <= WINDOW_SIZE; i++)
		{
			addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
			addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
		}
		for (int i = 0; i < NUM_OF_CAM; i++)
			addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
		if (ESTIMATE_TD)
		{
			addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
		}
	}
	else
	{
		// TODO：为什么这里要加这个判断条件
		if (Hprior_.rows() > 0)
		{
			vector2double();
			MargNewFrame();

			std::unordered_map<long, double *> addr_shift;
			for (int i = 0; i <= WINDOW_SIZE; i++)
			{
				if (i == WINDOW_SIZE - 1)
					continue;
				else if (i == WINDOW_SIZE)
				{
					addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
					addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
				}
				else
				{
					addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
					addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
				}
			}
			for (int i = 0; i < NUM_OF_CAM; i++)
				addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
			if (ESTIMATE_TD)
			{
				addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
			}
		}
	}
}


void Estimator::slideWindow()
{
	TicToc t_margin;
	if (marginalization_flag == MARGIN_OLD)
	{
		double t_0 = Headers[0];
		back_R0 = Rs[0];
		back_P0 = Ps[0];
		if (frame_count == WINDOW_SIZE)
		{
			for (int i = 0; i < WINDOW_SIZE; i++)
			{
				Rs[i].swap(Rs[i + 1]);

				std::swap(pre_integrations[i], pre_integrations[i + 1]);

				dt_buf[i].swap(dt_buf[i + 1]);
				linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
				angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

				Headers[i] = Headers[i + 1];
				Ps[i].swap(Ps[i + 1]);
				Vs[i].swap(Vs[i + 1]);
				Bas[i].swap(Bas[i + 1]);
				Bgs[i].swap(Bgs[i + 1]);
			}
			Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
			Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
			Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
			Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
			Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
			Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

			delete pre_integrations[WINDOW_SIZE];
			pre_integrations[WINDOW_SIZE] = new IntegrationBase{ acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE] };

			dt_buf[WINDOW_SIZE].clear();
			linear_acceleration_buf[WINDOW_SIZE].clear();
			angular_velocity_buf[WINDOW_SIZE].clear();

			if (true || solver_flag == INITIAL)
			{
				map<double, ImageFrame>::iterator it_0;
				it_0 = all_image_frame.find(t_0);
				delete it_0->second.pre_integration;
				it_0->second.pre_integration = nullptr;

				for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
				{
					if (it->second.pre_integration)
						delete it->second.pre_integration;
					it->second.pre_integration = NULL;
				}

				all_image_frame.erase(all_image_frame.begin(), it_0);
				all_image_frame.erase(t_0);
			}
			slideWindowOld();
		}
	}
	else
	{
		if (frame_count == WINDOW_SIZE)
		{
			for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
			{
				double tmp_dt = dt_buf[frame_count][i];
				Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
				Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

				pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

				dt_buf[frame_count - 1].push_back(tmp_dt);
				linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
				angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
			}

			Headers[frame_count - 1] = Headers[frame_count];
			Ps[frame_count - 1] = Ps[frame_count];
			Vs[frame_count - 1] = Vs[frame_count];
			Rs[frame_count - 1] = Rs[frame_count];
			Bas[frame_count - 1] = Bas[frame_count];
			Bgs[frame_count - 1] = Bgs[frame_count];

			delete pre_integrations[WINDOW_SIZE];
			pre_integrations[WINDOW_SIZE] = new IntegrationBase{ acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE] };

			dt_buf[WINDOW_SIZE].clear();
			linear_acceleration_buf[WINDOW_SIZE].clear();
			angular_velocity_buf[WINDOW_SIZE].clear();

			slideWindowNew();
		}
	}
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
	sum_of_front++;
	f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
	sum_of_back++;

	bool shift_depth = solver_flag == NON_LINEAR ? true : false;
	if (shift_depth)
	{
		Matrix3d R0, R1;
		Vector3d P0, P1;
		R0 = back_R0 * ric[0];
		R1 = Rs[0] * ric[0];
		P0 = back_P0 + back_R0 * tic[0];
		P1 = Ps[0] + Rs[0] * tic[0];
		f_manager.removeBackShiftDepth(R0, P0, R1, P1);
	}
	else
		f_manager.removeBack();
}