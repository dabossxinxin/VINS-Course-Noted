﻿#include <iostream>
#include <random>
#include "backend/problem.h"

#include <ceres/ceres.h>

using namespace std;
using namespace myslam::backend;

/**
* 曲线拟合模型的顶点：定义优化变量维度和数据类型
*/
class CurveFittingVertex : public Vertex
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		/*!
		*  @brief 曲线拟合顶点构造函数：直接通过基类构造函数构造
		*/
		CurveFittingVertex() : Vertex(3) {}

	/*!
	*  @brief 返回顶点的名称
	*  @return	std::string	顶点类型名称
	*/
	virtual std::string TypeInfo() const {
		return "abc";
	}
};

/**
* 曲线拟合模型的边：定义观测值的维度、类型、链接顶点类型
*/
class CurveFittingEdge : public Edge
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		/*!
		*  @brief 曲线拟合边构造函数
		*  @param[in]	x	x方向观测值
		*  @param[in]	y	y方向观测值
		*/
		CurveFittingEdge(double x, double y) :
		Edge(1, 1, std::vector<std::string>{"abc"}) {
		x_ = x;
		y_ = y;
	}

	/*!
	*  @brief 计算曲线拟合边残差=预测-观测
	*/
	virtual void ComputeResidual() override {
		Vec3 abc = verticies_[0]->Parameters();
		residual_(0) = std::exp(abc(0)*x_*x_ + abc(1)*x_ + abc(2)) - y_;
	}

	/*!
	*  @brief 计算曲线拟合边对顶点的雅可比
	*/
	virtual void ComputeJacobians() override {
		Vec3 abc = verticies_[0]->Parameters();
		double x_2 = x_*x_;
		double exp_y = std::exp(abc(0)*x_2 + abc(1)*x_ + abc(2));
		Eigen::Matrix<double, 1, 3> jaco_abc;
		jaco_abc << x_2 * exp_y, x_ * exp_y, exp_y;
		jacobians_[0] = jaco_abc;
	}

	/*!
	*  @brief 获取曲线拟合边的类型
	*/
	virtual std::string TypeInfo() const override {
		return "CurveFittingEdge";
	}
public:
	/*!< @brief 曲线拟合边观测值 */
	double x_, y_;
};

struct CurveFittingCost
{
	CurveFittingCost(double x, double y) :x_(x), y_(y){}

	template<typename T>
	bool operator()(const T* const abc, T* residual) const
	{
		residual[0] = T(y_) - ceres::exp(abc[0] * T(x_)*T(x_) + abc[1] * T(x_) + abc[2]);
		return true;
	}
	
	double x_, y_;
};

/* 
* 使用Ceres与使用自己实现的优化，在精度和效率两方面的比较 
* 精度：两者的精度保持一致
* 效率：同时使用狗腿法，Ceres效率是自己实现的方法的10倍
*/

int main() {
	/* 真实参数值&数据点数量&高斯噪声方差 */
	int N = 10000;
	double w_sigma = 1.;
	double a = 1.0, b = 2.0, c = 1.0;
	double abc[3] = { 0,0,0 };

	/* 带高斯噪声的数据生成器 */
	std::default_random_engine generator;
	std::normal_distribution<double> noise(0., w_sigma);

	/* 构建优化问题 */
	Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
	problem.SetThreadsNum(6);
	problem.SetNonLinearMethod(myslam::backend::Problem::NonLinearMethod::Dog_Leg);
	std::shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());
	/* 设置待优化顶点初始值 */
	vertex->SetParameters(Eigen::Vector3d(0., 0., 0.));
	/* 将待估计顶点加入到优化问题中 */
	problem.AddVertex(vertex);
	ceres::Problem ceresProblem;

	/* 构造N次观测 */
	for (int i = 0; i < N; ++i)
	{
		double x = i / 10000.;
		double n = noise(generator);
		double y = std::exp(a*x*x + b*x + c) + n;
		/* 构建边并将其加入到优化问题中 */
		LossFunction* loss = new HuberLoss(2.0);
		std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
		std::vector<std::shared_ptr<Vertex>> edge_vertex;
		edge_vertex.push_back(vertex);
		edge->SetVertex(edge_vertex);
		//edge->SetLossFunction(loss);
		problem.AddEdge(edge);

		ceresProblem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(
				new CurveFittingCost(x, y)),
			nullptr,
			abc
		);
	}

	/* 开始迭代求解优化问题 */
	problem.Solve(300);
	/* 获取优化后结果 */
	std::cout << "After Optimization,Parameters :" << std::endl;
	std::cout << vertex->Parameters().transpose() << std::endl;
	std::cout << "Ground Truth: " << std::endl;
	std::cout << "1.0,  2.0,  1.0" << std::endl << std::endl;
	
	/* 使用ceres拟合 */
	ceres::Solver::Options ceresOptions;
	ceresOptions.linear_solver_type = ceres::DENSE_QR;
	ceresOptions.minimizer_progress_to_stdout = true;
	ceresOptions.trust_region_strategy_type = ceres::DOGLEG;
	ceres::Solver::Summary ceresSummary;
	
	clock_t ceresStart = clock();
	ceres::Solve(ceresOptions, &ceresProblem, &ceresSummary);
	clock_t ceresEnd = clock();
	std::cout << "Ceres Optimization, Parameters :" << std::endl;
	std::cout << abc[0] << ", " << abc[1] << ", " << abc[2] << std::endl;
	std::cout << "Ceres Time: " << ceresEnd - ceresStart << "ms" << std::endl;

	/* 正常返回 */
	system("pause");
	return 0;
}