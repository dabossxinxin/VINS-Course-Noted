#include "System.h"
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_)
    :bStart_backend(true)
{
    string sConfig_file = sConfig_file_ + "simulate_config.yaml";

    cout << "1 System() sConfig_file: " << sConfig_file << endl;
    readParameters(sConfig_file);

    trackerData[0].readIntrinsicParameter(sConfig_file);

    estimator.setParameter();
    ofs_pose.open("./pose_output.txt",fstream::app | fstream::out);
    if(!ofs_pose.is_open())
    {
        cerr << "ofs_pose is not open" << endl;
    }
    // thread thd_RunBackend(&System::process,this);
    // thd_RunBackend.detach();
    cout << "2 System() end" << endl;
}

System::~System()
{
    bStart_backend = false;

    pangolin::Quit();
    
    m_buf.lock();
    while (!feature_buf.empty())
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();

    ofs_pose.close();
}

/*!
*  @brief 对传入VIO系统的图像进行处理
*  @detail 使用光流跟踪图像特征并将其放入feature_buf结构中
*  @param[in]	dStampSec	输入图像时间戳
*  @param[in]	img			输入图像数据
*/
void System::PubImageData(double dStampSec, Mat &img)
{
	/* 第一帧图像无法通过光流计算速度，直接返回并置init_feature=1 */
    if (!init_feature) {
        std::cout << "1 PubImageData skip the first detected feature, which doesn't contain optical flow speed" << std::endl;
        init_feature = 1;
        return;
    }
	
	/* 判断first_image_flag，并且将第一帧和最后一帧图像时间都设为当前图像时间 */
    if (first_image_flag) {
        std::cout << "2 PubImageData first_image_flag" << std::endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }
    
	/* 检测不稳定的图像流：当前图像时间戳远大于或小于系统最后一帧图像时间戳 */
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time)
    {
        std::cerr << "3 PubImageData image discontinue! reset the feature tracker!" << std::endl;
        first_image_flag = true;
        last_image_time = 0.;
        pub_count = 1;
        return;
    }

	/* 更新系统中最后一帧图像的时间戳为当前输入时间戳 */
    last_image_time = dStampSec;
    
	/* 控制图像发布的频率：保证VIO系统的稳定性和高效性 */
    if ((std::round)(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ) {
        PUB_THIS_FRAME = true;
        /* TODO */
        if ((std::abs)(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ) {
            first_image_time = dStampSec;
            pub_count = 0;
        }
    } else {
        PUB_THIS_FRAME = false;
    }

    /* 当前图像计算图像光流 */
    trackerData[0].readImage(img, dStampSec);

	/* TODO */
    for (unsigned int it = 0;; ++it) {
        bool completed = false;
        completed |= trackerData[0].updateID(it);

        if (!completed) break;
    }
	
	/* 若发布该图像数据，则将所有的特征点都保存到feature_buf中 */
    if (PUB_THIS_FRAME) 
	{
        pub_count++;
        std::shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        std::vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; ++i) 
		{
            auto &un_pts = trackerData[i].cur_un_pts;			// 跟踪得到的特征点
            auto &cur_pts = trackerData[i].cur_pts;				// 去畸变特征点
            auto &ids = trackerData[i].ids;						// 特征点对应ID
            auto &pts_velocity = trackerData[i].pts_velocity;	// 特征点对应速度
            for (unsigned int j = 0; j < ids.size(); ++j)
            {
				/* 取被跟踪此处大于1次的特征更新feature_points */
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Eigen::Vector3d(x, y, z));
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }

			/* 忽略第一张图像：第一张图像无法计算光流速度 */
            if (!init_pub) {
                std::cout << "4 PubImage init_pub skip the first image!" << std::endl;
                init_pub = 1;
            } else {
                m_buf.lock();
                feature_buf.push(feature_points);
                m_buf.unlock();
                con.notify_one();
            }
        }
    }

	/* 特征点显示时，被跟踪的次数越多，显示的颜色越深 */
#ifdef __windows__
    cv::Mat show_img;
	cv::cvtColor(img, show_img, CV_GRAY2RGB);
	if (SHOW_TRACK) {
		for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++) {
			double len = min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
			cv::circle(show_img, trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
		}
        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
		cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
	}
#endif    
}

/*!
*  @brief 对传入VIO系统的图像进行处理：模拟数据
*  @param[in]	dStampSec		输入图像时间戳
*  @param[in]	featurePoints	输入图像特征
*/
void System::PubImageData(double dStampSec, const std::vector<cv::Point2f>& featurePoints)
{
	if (!init_feature) {
		std::cout << "1 PubImageData skip the first feature, which doesn't contain optical flow speed" << std::endl;
		init_feature = 1;
	}

	if (first_image_flag) {
		std::cout << "2 PubImageData first_image_flag" << std::endl;
		first_image_flag = false;
		first_image_time = dStampSec;
		last_image_time = dStampSec;
		return;
	}

	if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time) {
		std::cerr << "3 PubImageData image discontinue! reset the feature tracker!" << std::endl;
		first_image_flag = true;
		last_image_time = 0;
		pub_count = 1;
		return;
	}
	last_image_time = dStampSec;
	PUB_THIS_FRAME = true;
	
	if (PUB_THIS_FRAME) {
		pub_count++;
		std::shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
		feature_points->header = dStampSec;
		std::vector<std::set<int>> hash_ids(NUM_OF_CAM);
		for (int iti = 0; iti < NUM_OF_CAM; ++iti) {
			for (int itj = 0; itj < featurePoints.size(); ++itj) {
				int p_id = itj;
				hash_ids[iti].insert(p_id);
				double x = featurePoints[itj].x;
				double y = featurePoints[itj].y;
				double z = 1.0;
				feature_points->points.push_back(Eigen::Vector3d(x, y, z));
				feature_points->id_of_point.push_back(p_id*NUM_OF_CAM + iti);

				cv::Point2f pixel_point;
				pixel_point.x = 460 * x + 255;
				pixel_point.y = 460 * y + 255;

				feature_points->u_of_point.push_back(pixel_point.x);
				feature_points->v_of_point.push_back(pixel_point.y);
				
				feature_points->velocity_x_of_point.push_back(0.);
				feature_points->velocity_y_of_point.push_back(0.);
			}

			if (!init_pub) {
				std::cout << "4 PubImage init_pub skip the first image!" << std::endl;
				init_pub = 1;
			}
			else {
				m_buf.lock();
				feature_buf.push(feature_points);
				m_buf.unlock();
				con.notify_one();
			}
		}
	}
}

/*!
*  @brief VIO系统中将IMU数据与相机数据对齐：获取时间对齐的相机与IMU测量数据
*  @detail [1]:IMU或者相机帧的buf为空，measurements返回空值
*		   [2]:IMU最新的时间戳imu_buf.back()小于最旧的图像帧feature_buf.front()，等待IMU刷新
*		   [3]:IMU最旧的时间戳imu_buf.front()大于图像帧最旧的图像帧feature_buf.front(),等待IMU刷新
*		   [4]:理想情况:IMU最旧的在图像帧最旧的之前，保证有足够的IMU数据
*  @return	std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>>	对齐后的IMU数据与相机数据
*/
std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> measurements;
	/* 对IMU和图像数据进行对齐组合，直到把缓存中的图像数据或者IMU数据读完，才能跳出函数返回数据 */
    while (true) {
		/* 检查imu_buf是否为空 */
		if (imu_buf.empty()) {
			//std::cerr << "imu_buf.empty()" << std::endl;
			return measurements;
		}
		/* 检查feature_buf是否为空 */
        if (feature_buf.empty()) {
            //std::cerr << "feature_buf.empty()" << std::endl;
            return measurements;
        }
		/* IMU中最新的数据时间戳小于图像帧中最旧的数据，此时等待IMU数据刷新 */
        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator.td)) {
            std::cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                << sum_of_wait << std::endl;
            sum_of_wait++;
            return measurements;
        }
		/* IMU中最旧的数据时间戳大于图像帧中最旧的数据，此时最旧的图像帧数据需要删除 */
        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator.td)) {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
		/* 最理想的情况是最旧的IMU信息比最旧的图像信息还旧 */
        std::vector<ImuConstPtr> IMUs;
        while (imu_buf.front()->header < img_msg->header + estimator.td) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        /* 多放一组IMU信息进来，后面便于使用插值将时间戳完全对齐 */
		/* 由于这组IMU信息没有pop，因此当前图形帧与下一图像帧会共用这组数据 */
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()){
            std::cerr << "no imu between two image" << std::endl;
        }
        // std::cout << "1 getMeasurements img t: " << fixed << img_msg->header
        //			<< " imu begin: "<< IMUs.front()->header 
        //			<< " end: " << IMUs.back()->header
        //			<< std::endl;
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/*!
*  @brief 对传入VIO系统的IMU数据进行处理
*  @detail 将传入VIO系统中的数据放到结构imu_buf中
*  @param[in]	dStamp	输入IMU数据时间戳
*  @param[in]	vGyr	输入IMU数据角速度
*  @param[in]	vAcc	输入IMU数据加速度
*/
void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
						const Eigen::Vector3d &vAcc)
{
	/* 通过传入数据初始化一个IMU_MSG */
    std::shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;
	
	/* 检查传入的IMU数据时间戳是否正常 */
	/*std::cout << "last_imu_t: " << last_imu_t << " "
		<< "dStampSec: " << dStampSec << std::endl;*/
    if (dStampSec <= last_imu_t) {
        std::cerr << "Imu Message In Disorder!" << std::endl;
        return;
    }
    last_imu_t = dStampSec;
    /*std::cout << "1 PubImuData t: " << fixed << imu_msg->header
			<< " acc: " << imu_msg->linear_acceleration.transpose()
			<< " gyr: " << imu_msg->angular_velocity.transpose() << std::endl;*/
    m_buf.lock();
    imu_buf.push(imu_msg);
    /*std::cout << "1 PubImuData t: " << fixed << imu_msg->header 
			<< " imu_buf size:" << imu_buf.size() << std::endl;*/
    m_buf.unlock();
    con.notify_one();
}

// thread: visual-inertial odometry
void System::ProcessBackEnd()
{
    cout << "1 ProcessBackEnd Start" << endl;
    while (bStart_backend)
    {
        /* 获取时间戳对齐的相机&IMU测量数据 */
        std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> measurements;
        std::unique_lock<mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        if( measurements.size() > 1){
        cout << "1 getMeasurements size: " << measurements.size() 
            << " imu sizes: " << measurements[0].first.size()
            << " feature_buf size: " <<  feature_buf.size()
            << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();

		/* 针对对齐的测量数据，估计系统状态量 */
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
			/* 将IMU数据送入Estimator类中进行预积分计算帧间位置、姿态、速度、偏置量 */
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;
                if (t <= img_t) {
					/* 将小于等于图像时间戳的IMU数据全部中值积分预积分 */
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    estimator.processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                    //printf("1 BackEnd imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                } else {
					/* 对齐的图像数据与IMU数据之间时间戳仍会存在差距，使用线性插值校正 */
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            /*std::cout << "processing vision data with stamp:" << img_msg->header 
					<< " img_msg->points.size: "<< img_msg->points.size() << std::endl;*/

            // TicToc t_s;
			/*  */
            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
				int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x();		// 去畸变后x方向像素坐标
                double y = img_msg->points[i].y();		// 去畸变后y方向像素坐标
                double z = img_msg->points[i].z();		// 1.0
                double p_u = img_msg->u_of_point[i];	// 特征点x方向像素坐标
                double p_v = img_msg->v_of_point[i];	// 特征点y方向像素坐标
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            estimator.processImage(image, img_msg->header);


            
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                p_wi = estimator.Ps[WINDOW_SIZE];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator.Headers[WINDOW_SIZE];
                std::cout << "1 BackEnd processImage dt: " << fixed 
					<< t_processImage.toc() 
					<< " stamp: " <<  dStamp 
					<< " p_wi: " << p_wi.transpose() 
					<< std::endl;

				/* 以TUM格式保存数据，方便轨迹精度评价 */
				ofs_pose.precision(9);
				ofs_pose << dStamp << " ";
				ofs_pose.precision(5);
                ofs_pose << p_wi(0) << " "
						<< p_wi(1) << " "
						<< p_wi(2) << " "
						<< q_wi.x() << " "
						<< q_wi.y() << " "
						<< q_wi.z() << " "
						<< q_wi.w() << std::endl;
            }
        }
        m_estimator.unlock();
    }
}

void System::Draw() 
{   
    /* 创建Pangolin显示界面：显示VIO系统轨迹 */
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) 
	{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
         
        /* 绘制轨迹 */
		if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
		{
			glColor3f(0, 0, 0);
			glLineWidth(2);
			glBegin(GL_LINES);
			int nPath_size = vPath_to_draw.size();
			if (nPath_size > 0) 
			{
				for (int i = 0; i < nPath_size - 1; ++i)
				{
					glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
					glVertex3f(vPath_to_draw[i + 1].x(), vPath_to_draw[i + 1].y(), vPath_to_draw[i + 1].z());
				}
			}
			glEnd();
		}
        
        /* 绘制离散轨迹点 */
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        cv::waitKey(50);   // sleep 50 ms
    }

#ifdef __APPLE__
void System::InitDrawGL() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
}

void System::DrawGLFrame() 
{  

    if (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
            
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
#endif

}
