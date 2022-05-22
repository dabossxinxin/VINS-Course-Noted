#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
std::string sConfig_path = "D:\\Code\\VINS-Course-Noted\\config\\";

std::shared_ptr<System> pSystem;

/*!
*  @brief 系统读取IMU数据
*/
void PubImuData()
{
	/* 设置IMU数据句柄 */
	std::string sImu_data_file = "D:\\Code\\VINS-Course-Noted\\config\\simulate\\imu_pose_noise.txt";
	std::cout << "1 PubImuData Start sImu_data_file: " << sImu_data_file << std::endl;
	std::ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open()) {
		std::cerr << "Failed To Open Imu File! " << sImu_data_file << std::endl;
		return;
	}

	/* 读取IMU数据 */
	double tmp = 0.;
	std::string sImu_line;
	double dStampNSec = 0.0;
	Eigen::Vector3d vAcc;
	Eigen::Vector3d vGyr;
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) {
		/* 读取IMU数据 */
		std::istringstream ssImuData(sImu_line);
		ssImuData >> dStampNSec;
		for (int it = 0; it < 7; ++it) ssImuData >> tmp;
		ssImuData >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		/* 将IMU数据加入到系统中 */
		pSystem->PubImuData(dStampNSec, vGyr, vAcc);
		//usleep(5000*nDelayTimes);
		cv::waitKey(5 * nDelayTimes);
	}
	/* 关闭IMU句柄 */
	fsImu.close();
}

/*!
*  @brief 系统读取Image数据
*/
void PubImageData()
{
	/* 设置图像数据句柄 */
	std::string sImage_file = "D:\\Code\\VINS-Course-Noted\\config\\simulate\\cam_pose.txt";
	std::cout << "1 PubImageData Start sImage_file: " << sImage_file << std::endl;
	std::ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open()) {
		std::cerr << "Failed To Open Image File! " << sImage_file << std::endl;
		return;
	}

	/* 读取图像数据 */
	std::string sImage_line;
	double dStampNSec;
	std::string sImgFileName;
	int n = 0;
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
		/* 获取图像读取路径 */
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec;
		std::string all_points_file_name =
			"D:\\Code\\VINS-Course-Noted\\config\\simulate\\keyframe\\all_points_" + std::to_string(n) + ".txt";
		
		std::vector<cv::Point2f> featurePoints;
		std::ifstream fIn;
		fIn.open(all_points_file_name);
		while (!fIn.eof()) {
			std::string s;
			std::getline(fIn, s);
			if (!s.empty()) {
				std::stringstream ss;
				ss << s;

				double tmp = 0.;
				for (int it = 0; it < 4;++it) ss >> tmp;
				
				double px, py;
				ss >> px >> py;
				cv::Point2f pt(px, py);
				featurePoints.push_back(pt);
			}
		}
		/* 将图像数据加入到VIO系统中 */
		pSystem->PubImageData(dStampNSec, featurePoints);
		//usleep(50000*nDelayTimes);
		cv::waitKey(50 * nDelayTimes);
		n++;
	}
	/* 关闭图像句柄 */
	fsImage.close();
}

#ifdef __APPLE__
void DrawIMGandGLinMainThrd() {
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;

	pSystem->InitDrawGL();
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		//pSystem->PubImageData(dStampNSec / 1e9, img);
		cv::Mat show_img;
		cv::cvtColor(img, show_img, CV_GRAY2RGB);
		if (SHOW_TRACK)
		{
			for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size(); j++)
			{
				double len = min(1.0, 1.0 *  pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
				cv::circle(show_img, pSystem->trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
			}

			cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
			cv::imshow("IMAGE", show_img);
			// cv::waitKey(1);
		}

		pSystem->DrawGLFrame();
		usleep(50000 * nDelayTimes);
	}
	fsImage.close();

}
#endif

int main(int argc, char **argv)
{
	sConfig_path = "D:\\Code\\VINS-Course-Noted\\config\\";
	pSystem.reset(new System(sConfig_path));

	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
	std::thread thd_PubImuData(PubImuData);
	std::thread thd_PubImageData(PubImageData);

#ifdef __linux__
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __windows__
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif

	thd_PubImuData.join();
	thd_PubImageData.join();
	thd_BackEnd.join();
	thd_Draw.join();

	std::cout << "Main End...See You ..." << std::endl;
	system("pause");
	return 0;
}
