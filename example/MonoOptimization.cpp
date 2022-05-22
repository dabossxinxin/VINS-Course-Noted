#include <iostream>
#include <random>
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"

using namespace std;
using namespace myslam::backend;

#define M_PI       3.14159265358979323846   // pi

struct Frame {
	Frame(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),twc(t),qwc(R) {};
	Eigen::Matrix3d		Rwc;
	Eigen::Quaterniond	qwc;
	Eigen::Vector3d		twc;
	std::unordered_map<int, Eigen::Vector3d> featurePerId;
};

void GetSimDataInWorldFrame(std::vector<Frame>& cameraPoses, std::vector<Eigen::Vector3d>& points) {
	int featureNums = 2000;
	int poseNums = 10;
	
	double radius = 8;
	for (int it = 0; it < poseNums; ++it) {
		double theta = it * 2 * M_PI / (poseNums * 4);
		Eigen::Matrix3d R;
		R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
		Eigen::Vector3d t = Eigen::Vector3d(radius*cos(theta) - radius, radius*sin(theta), 1 * sin(2 * theta));
		cameraPoses.push_back(Frame(R, t));
	}

	std::default_random_engine generator;
	std::normal_distribution<double> noise_pdf(0., 1./100);
	for (int it = 0; it < featureNums; ++it) {
		std::uniform_real_distribution<double> xy_rand(-4, 4);
		std::uniform_real_distribution<double> z_rand(4., 8.);

		Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
		points.push_back(Pw);

		for (int i = 0; i < poseNums; ++i) {
			Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose()*(Pw - cameraPoses[i].twc);
			Pc = Pc / Pc.z();               // 归一化图像平面
			Pc[0] += noise_pdf(generator);  // 高斯噪声
			Pc[1] += noise_pdf(generator);
			cameraPoses[i].featurePerId.insert(std::make_pair(it, Pc));
		}
	}
}

int main() {
	// 准备数据
	std::vector<Frame> cameras;
	std::vector<Eigen::Vector3d> points;
	GetSimDataInWorldFrame(cameras, points);
	Eigen::Quaterniond qic(1, 0, 0, 0);
	Eigen::Vector3d tic(0, 0, 0);

	Problem problem(Problem::ProblemType::SLAM_PROBLEM);

	/* 向优化问题中添加优化顶点 */
	std::vector<std::shared_ptr<VertexPose>> vertexCams_vec;
	for (int it = 0; it < cameras.size(); ++it) {
		std::shared_ptr<VertexPose> vertexCam(new VertexPose());
		Eigen::VectorXd pose(7);
		pose << cameras[it].twc, cameras[it].qwc.x(), cameras[it].qwc.y(), cameras[it].qwc.z(), cameras[it].qwc.w();
		vertexCam->SetParameters(pose);

		if (it < 2) {
			vertexCam->SetFixed();
		}
		/* 向问题中添加相机顶点 */
		problem.AddVertex(vertexCam);
		vertexCams_vec.push_back(vertexCam);
	}
	
	std::default_random_engine generator;
	std::normal_distribution<double> noise_pdf(0., 1.);
	double noise = 0.;
	std::vector<double> noise_invd;

	MatXX information(2, 2);
	information.setIdentity();
	information *= 1.5;

	/* 向优化问题中添加边 */
	std::vector<std::shared_ptr<VertexInverseDepth>> allPoints;
	for (int iti = 0; iti < points.size(); ++iti) {
		Eigen::Vector3d Pw = points[iti];
		Eigen::Vector3d Pc = cameras[0].Rwc.transpose()*(Pw - cameras[0].twc);
		noise = noise_pdf(generator);
		double inverse_depth = 1. / (Pc.z() + noise);
		noise_invd.push_back(inverse_depth);

		/* 向问题中添加路标点 */
		std::shared_ptr<VertexInverseDepth>
			vertexPoint(new VertexInverseDepth());
		VecX inv_d(1);
		inv_d << inverse_depth;
		vertexPoint->SetParameters(inv_d);
		problem.AddVertex(vertexPoint);
		allPoints.push_back(vertexPoint);

		for (int itj = 1; itj < cameras.size(); ++itj) {
			Eigen::Vector3d pt_i = cameras[0].featurePerId.find(iti)->second;
			Eigen::Vector3d pt_j = cameras[itj].featurePerId.find(iti)->second;

			std::shared_ptr<EdgeReprojectionICFixed> 
				edge(new EdgeReprojectionICFixed(pt_i, pt_j));

			LossFunction* huber = new HuberLoss(1.5);
			
			std::vector<std::shared_ptr<Vertex>> edge_vertex;
			edge_vertex.push_back(vertexPoint);
			edge_vertex.push_back(vertexCams_vec[0]);
			edge_vertex.push_back(vertexCams_vec[itj]);
			edge->SetVertex(edge_vertex);
			//edge->SetLossFunction(huber);
			edge->SetInformation(information);
			
			/* 向问题中添加边 */
			problem.AddEdge(edge);
		}
	}
	
	problem.Solve(5);
	
	std::cout << "\nCompare MonoBA Results After Opt..." << std::endl;
	for (size_t k = 0; k < allPoints.size(); ++k) {
		std::cout << "After Opt, Point " << k << " : GroundTruth " << 1. / points[k].z() << " ,Noise "
			<< noise_invd[k] << " ,Opt " << allPoints[k]->Parameters() << std::endl;
	}
	std::cout << "------------ Pose Translation ----------------" << std::endl;
	for (int i = 0; i < vertexCams_vec.size(); ++i) {
		std::cout << "Translation After Opt: " << i << " :" << vertexCams_vec[i]->Parameters().head(3).transpose() << " || GroundTruth: " << cameras[i].twc.transpose() << std::endl;
	}
	std::cout << "--------------Eular Angle ----------------" << std::endl;
	for (int it = 0; it < vertexCams_vec.size(); ++it) {
		Eigen::VectorXd pose = vertexCams_vec[it]->Parameters();
		Qd Qi(pose[6], pose[3], pose[4], pose[5]);
		Eigen::Vector3d eularAngleOpt = Qi.toRotationMatrix().eulerAngles(2, 1, 0);
		Eigen::Vector3d eularAngle = cameras[it].Rwc.eulerAngles(2, 1, 0);
		std::cout << "Eular Angle After Opt: " << "it" << " :" << eularAngleOpt.transpose() << " || GroundTruth: " << eularAngle.transpose() << std::endl;
	}
	
	problem.TestMarginalize();
	system("pause");
	return 0;
}