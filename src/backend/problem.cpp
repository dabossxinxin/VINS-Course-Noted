#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
	std::ofstream f(name.c_str());
	f << matrix.format(CSVFormat);
}

namespace myslam {
	namespace backend {

		/*!
		*  @brief 打印调试信息
		*/
		void Problem::LogoutVectorSize()
		{
			/*LOG(INFO) << "1 problem::LogoutVectorSize verticies_:"
				<< verticies_.size()
				<< " edges:"
				<< edges_.size();*/
		}

		/*!
		*  @brief 优化问题构造函数
		*  @detail 初始化问题类型，迭代中止阈值
		*  @param[in]   problemType 优化问题类型
		*/
		Problem::Problem(ProblemType problemType) :
			problemType_(problemType), deltaX_norm_threshold_(1e-6),
			delta_chi_threshold_(1e-6),
			non_linear_method_(NonLinearMethod::Levenberge_Marquardt)
		{
			std::cout << "Optimize Method: ";
			if (non_linear_method_ == NonLinearMethod::Levenberge_Marquardt)
				std::cout << "Levenberge_Marquardt" << std::endl;
			else if (non_linear_method_ == NonLinearMethod::Dog_Leg)
				std::cout << "Dog_Leg" << std::endl;
			LogoutVectorSize();
			verticies_marg_.clear();
		}

		/*!
		*  @brief 优化问题析构函数
		*  @param[in]   problemType 优化问题类型
		*/
		Problem::~Problem()
		{
			std::cout << "Problem Is Deleted" << std::endl;
			global_vertex_id = 0;
		}

		/*!
		*  @brief 向优化问题中添加待待优化量[节点]
		*  @param[in]	vertex	待优化顶点
		*  @retuan		bool	是否成功添加顶点
		*/
		bool Problem::AddVertex(std::shared_ptr<Vertex> vertex)
		{
			/* 判断节点vertex是否已经添加到优化问题中 */
			if (verticies_.find(vertex->Id()) != verticies_.end()) {
				std::cout << "Vertex " << vertex->Id() << " Has Been Added Before" << std::endl;
				return false;
			}
			else {
				verticies_.insert(std::pair<unsigned long, std::shared_ptr<Vertex>>(vertex->Id(), vertex));
			}
			/* SLAM问题中，添加vertex后需要扩充Hessian矩阵维度 */
			if (problemType_ == ProblemType::SLAM_PROBLEM) {
				if (IsPoseVertex(vertex)) {
					ResizePoseHessiansWhenAddingPose(vertex);
				}
			}
			return true;
		}

		/*!
		*  @brief SLAM问题中设置新加入顶点的顺序
		*  @param[in]   v   新加入SLAM问题中的顶点
		*/
		void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v)
		{
			/* SLAM问题中添加Pose顶点 */
			if (IsPoseVertex(v)) {
				v->SetOrderingId(ordering_poses_);
				idx_pose_vertices_.insert(std::pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
				ordering_poses_ += v->LocalDimension();
				/* SLAM问题中添加LandMark顶点 */
			}
			else if (IsLandmarkVertex(v)) {
				v->SetOrderingId(ordering_landmarks_);
				ordering_landmarks_ += v->LocalDimension();
				idx_landmark_vertices_.insert(std::pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
			}
		}

		/*!
		*  @brief 添加顶点后，需要调整先验Hessian&先验残差的大小
		*  @param[in]   v   新加入的顶点
		*/
		void Problem::ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v)
		{
			/* 扩充先验Hessian矩阵&先验残差矩阵的维度 */
			int size = H_prior_.rows() + v->LocalDimension();
			H_prior_.conservativeResize(size, size);
			b_prior_.conservativeResize(size);

			/* 以上扩充维度的部分全部设置为零 */
			b_prior_.tail(v->LocalDimension()).setZero();
			H_prior_.rightCols(v->LocalDimension()).setZero();
			H_prior_.bottomRows(v->LocalDimension()).setZero();
		}

		/*!
		*  @brief 按照既定要求扩充先验信息矩阵及残差矩阵的维度
		*  @detail 边缘化掉某些优化量后，Hessian矩阵和b矩阵维度会降低，
		*          在新的状态进来后，先验Hessian矩阵和b矩阵维度需要和，
		*          新的滑窗中状态量的维度保持一致，因此要扩充先验维度；
		*  @param[in]   dim   需扩充的维度
		*/
		void Problem::ExtendHessiansPriorSize(int dim)
		{
			/* 扩充先验Hessian矩阵&先验残差矩阵的维度 */
			int size = H_prior_.rows() + dim;
			H_prior_.conservativeResize(size, size);
			b_prior_.conservativeResize(size);
			/* 以上扩充维度的部分全部设置为零 */
			b_prior_.tail(dim).setZero();
			H_prior_.rightCols(dim).setZero();
			H_prior_.bottomRows(dim).setZero();
		}

		/*!
		*  @brief 判断顶点是否为Pose顶点
		*  @param[in]   v       输入顶点
		*  @return      bool    输入节点是否为Pose节点
		*/
		bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v)
		{
			std::string type = v->TypeInfo();
			return type == std::string("VertexPose") ||
				type == std::string("VertexSpeedBias");
		}

		/*!
		*  @brief 判断顶点是否为Pose顶点
		*  @param[in]   v       输入顶点
		*  @return      bool    输入节点是否为Pose节点
		*/
		bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v)
		{
			std::string type = v->TypeInfo();
			return type == std::string("VertexPointXYZ") ||
				type == std::string("VertexInverseDepth");
		}

		/*!
		*  @brief 向优化问题中添加边
		*  @param[in]	vertex	待添加边
		*  @return		bool	是否成功添加边
		*/
		bool Problem::AddEdge(std::shared_ptr<Edge> edge)
		{
			/* 判断边edge是否已经添加到优化问题中 */
			if (edges_.find(edge->Id()) == edges_.end()) {
				edges_.insert(std::pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
			}
			else {
				std::cerr << "Edge " << edge->Id() << " Has Been Added Before!";
				return false;
			}
			/* 更新vertexToEdge_，便于由vertex查询edge */
			for (auto &vertex : edge->Verticies()) {
				vertexToEdge_.insert(std::pair<ulong, std::shared_ptr<Edge>>(vertex->Id(), edge));
			}
			return true;
		}

		/*!
		*  @brief 获取指定顶点相链接的边
		*  @param[in]   vertex                          需获取链接边的顶点
		*  @return  std::vector<std::shared_ptr<Edge>>  输入顶点链接的边
		*/
		std::vector<std::shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex)
		{
			std::vector<std::shared_ptr<Edge>> edges;
			auto range = vertexToEdge_.equal_range(vertex->Id());
			for (auto iter = range.first; iter != range.second; ++iter)
			{
				/* 在优化问题的所有边中寻找vertex节点对应的边 */
				if (edges_.find(iter->second->Id()) == edges_.end())
				{
					continue;
				}
				edges.emplace_back(iter->second);
			}
			return edges;
		}

		/*!
		*  @brief 移除优化问题中的优化变量
		*  @param[in]	vertex	待优化顶点
		*  @return		bool	是否成功移除顶点
		*/
		bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex)
		{
			/* 查询当前待删除的节点是否存在于优化问题的所有节点中 */
			if (verticies_.find(vertex->Id()) == verticies_.end()) {
				std::cout << "The Vertex " << vertex->Id() << " Is Not In The Problem!" << std::endl;
				return false;
			}
			/* 删除节点vertex对应的edge */
			std::vector<std::shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
			for (size_t i = 0; i < remove_edges.size(); i++) {
				RemoveEdge(remove_edges[i]);
			}
			/* 更新数据结构idx_pose_verticies和idx_landmark_vertices */
			if (IsPoseVertex(vertex)) {
				idx_pose_vertices_.erase(vertex->Id());
			}
			else {
				idx_landmark_vertices_.erase(vertex->Id());
			}
			/* 用于debug */
			vertex->SetOrderingId(-1);
			/* 更新数据结构vertices_和vertexToEdge_ */
			verticies_.erase(vertex->Id());
			vertexToEdge_.erase(vertex->Id());
			/* 正常返回 */
			return true;
		}

		/*!
		*  @brief 移除优化问题中的边
		*  @param[in]	vertex	待移除边
		*  @return		bool	是否成功移除边
		*/
		bool Problem::RemoveEdge(std::shared_ptr<Edge> edge)
		{
			/* 检查待删除的边是否在优化问题的边集合中 */
			if (edges_.find(edge->Id()) == edges_.end()) {
				std::cout << "The Edge " << edge->Id() << " Is Not In The Problem!" << std::endl;
				return false;
			}
			/* 更新数据结构edges_ */
			edges_.erase(edge->Id());
			/* 正常返回 */
			return true;
		}

		/*!
		*  @brief 求解当前的优化问题
		*  @param[in]	iterations  非线性优化迭代次数
		*  @return      bool        非线性优化问题是否成功收敛
		*/
		bool Problem::Solve(int iterations)
		{
			/* 检查优化问题中是否具有顶点元素以及边元素 */
			if (edges_.size() == 0 || verticies_.size() == 0)
			{
				std::cerr << "\nCannot Solve Problem Without Edges Or Verticies" << std::endl;
				return false;
			}
			/* 记录LM求解的时间 */
			TicToc t_solve;
			/* 统计优化变量的维度：为构建Hessian矩阵做准备 */
			SetOrdering();
			/* 遍历edge_：构建Hessian矩阵 */
			MakeHessian();
			/* LM初始化 */
			ComputeLambdaInitLM();
			/* LM开始进行 */
			bool stop = false;
			int iter = 0;
			double last_chi_ = 1e+20;
			while (!stop && (iter < iterations))
			{
				/* 打印优化的关键信息 */
				std::cout << "iter: " << iter
					<< " , chi= " << currentChi_
					<< " , lambda= " << currentLambda_
					<< " , radius= " << dogleg_radius_
					<< std::endl;

				int false_cnt = 0;
				bool oneStepSuccess = false;

				/* 不断尝试Lambda：直到成功迭代一步 */
				while (!oneStepSuccess && false_cnt < 10)
				{
					/* 求解delta_x */
					SolveLinearSystem();

					/* 优化退出条件1：delata_x很小，那么退出迭代 */
					if (this->delta_x_.squaredNorm() <= deltaX_norm_threshold_) {
						stop = true;
					}

					/* 更新状态量delta_x */
					UpdateStates();

					/* 判断当前迭代是否使残差下降，并更新Lambda */
					oneStepSuccess = IsGoodStepInLM();

					if (oneStepSuccess)
					{
						/* 在新的线性化点，构建Hessian矩阵 */
						MakeHessian();
						false_cnt = 0;
					}
					else {
						false_cnt++;
						/* 误差并没有下降，回滚到上一步状态量 */
						RollbackStates();
					}
				}
				iter++;

				/*优化退出条件3：与上一步残差相比，已经不怎么下降了*/
				if (last_chi_ - currentChi_ < delta_chi_threshold_)
				{
					stop = true;
				}
				last_chi_ = currentChi_;
			}
			std::cout << "\tSolve Problem Cost: " << t_solve.toc() << " ms" << std::endl;
			std::cout << "\tMake Hessian Cost: " << t_hessian_cost_ << " ms" << std::endl;
			t_hessian_cost_ = 0.;
			return true;
		}

		/*!
		*  @brief Solve的实现：求解通用非线性优化问题
		*  @param[in]	iterations  非线性优化迭代次数
		*  @return      bool        非线性优化问题是否成功收敛
		*/
		bool Problem::SolveGenericProblem(int iterations) {
			return true;
		}

		/*!
		*  @brief 设置优化问题中各个顶点的ID
		*  @detail 遍历加入优化问题中的顶点，计算pose数量以及landmark数量，
		*          并且ordering_generic=ordering_poses+ordering_landmarks
		*/
		void Problem::SetOrdering()
		{
			/* 初始化维度 */
			ordering_poses_ = 0;
			ordering_generic_ = 0;
			ordering_landmarks_ = 0;
			/* 统计优化问题中顶点维度 */
			for (auto vertex : verticies_)
			{
				ordering_generic_ += vertex.second->LocalDimension();
				/* 如果是SLAM问题：需要分别统计Pose与LandMark维度 */
				if (problemType_ == ProblemType::SLAM_PROBLEM)
				{
					AddOrderingSLAM(vertex.second);
				}
			}
			/* 此处需要把LandMark的Ordering加上Pose的数量：保证LandMark顺序在后Pose在前 */
			if (problemType_ == ProblemType::SLAM_PROBLEM)
			{
				ulong all_pose_dimension = ordering_poses_;
				for (auto landmarkVertex : idx_landmark_vertices_)
				{
					landmarkVertex.second->SetOrderingId(
						landmarkVertex.second->OrderingId() + all_pose_dimension
					);
				}
			}
		}

		/*!
		*  @brief 检查Ordering是否正确
		*  @detail 只有在SLAM问题中需要检查顶点ID是否成功设置；
		*          检查方法为将遍历所有顶点，并且统计pose顶点与landmark的顶点数量
		*/
		bool Problem::CheckOrdering()
		{
			if (problemType_ == ProblemType::SLAM_PROBLEM)
			{
				int current_ordering = 0;
				/* 检查Pose节点的顺序是否正确 */
				for (auto v : idx_pose_vertices_)
				{
					assert(v.second->OrderingId() == current_ordering);
					current_ordering += v.second->LocalDimension();
				}
				/* 检查LandMark节点的顺序是否正确 */
				for (auto v : idx_landmark_vertices_)
				{
					assert(v.second->OrderingId() == current_ordering);
					current_ordering += v.second->LocalDimension();
				}
			}
			return true;
		}

		/*!
		*  @brief 构造大Hessian矩阵
		*  @detail 1：遍历所有edge，构造关于优化变量的Hessian矩阵
		*          2：若有先验信息，将先验信息与所构造的Hessian相加
		*/
		void Problem::MakeHessian()
		{
			/* 记录Hessian矩阵的构造时间 */
			TicToc t_h;

			/* 1 通用问题：ordering_generic_=odering_generic_ */
			/* 2 SLAM问题：ordering_generic_=ordering_pose_+ordering_landmark_ */
			ulong size = ordering_generic_;
			MatXX H(MatXX::Zero(size, size));
			VecX b(VecX::Zero(size));

			/* 遍历edge_：计算Hessian矩阵 */
#ifdef USE_OPENMP
//#pragma omp parallel for
#endif
			for (auto &edge : edges_)
			{
				edge.second->ComputeResidual();
				edge.second->ComputeJacobians();

				auto jacobians = edge.second->Jacobians();
				auto verticies = edge.second->Verticies();
				assert(jacobians.size() == verticies.size());
				for (size_t i = 0; i < verticies.size(); ++i)
				{
					auto v_i = verticies[i];

					/* 若顶点固定，则对应雅可比为0 */
					if (v_i->IsFixed()) continue;

					auto jacobian_i = jacobians[i];
					ulong index_i = v_i->OrderingId();
					ulong dim_i = v_i->LocalDimension();

					/* 为当前边添加鲁棒核函数 */
					double drho = 1.0;
					MatXX robustInfo(edge.second->Information().rows(), edge.second->Information().cols());
					edge.second->RobustInfo(drho, robustInfo);

					MatXX JtW = jacobian_i.transpose() * robustInfo;
					for (size_t j = i; j < verticies.size(); ++j)
					{
						auto v_j = verticies[j];

						if (v_j->IsFixed()) continue;

						auto jacobian_j = jacobians[j];
						ulong index_j = v_j->OrderingId();
						ulong dim_j = v_j->LocalDimension();

						assert(v_j->OrderingId() != -1);
						MatXX hessian = JtW * jacobian_j;

						H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
						if (j != i) {
							H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
						}
					}
					b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
				}
			}
			Hessian_ = H;
			b_ = b;
			t_hessian_cost_ += t_h.toc();

			if (H_prior_.rows() > 0)
			{
				MatXX H_prior_tmp = H_prior_;
				VecX b_prior_tmp = b_prior_;

				/* 将顶点集合中被固定的顶点先验设为0：固定顶点不变化不需要雅可比 */
				for (auto vertex : verticies_)
				{
					if (IsPoseVertex(vertex.second) && vertex.second->IsFixed())
					{
						int idx = vertex.second->OrderingId();
						int dim = vertex.second->LocalDimension();
						H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
						H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
						b_prior_tmp.segment(idx, dim).setZero();
					}
				}

				/* 先验中仅仅包含Pose先验，而没有LandMark先验 */
				Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
				b_.head(ordering_poses_) += b_prior_tmp;
			}

			delta_x_ = VecX::Zero(size);
		}

		/*!
		*  @brief 求解线性方程
		*  @detail 求解线性方程时，根据问题的性质提供了两种求解方法
		*          通用问题：直接求解Hessian矩阵的逆求解优化量增量
		*          SLAM问题：边缘化掉landmark后，求解剩下的矩阵逆求解优化量增量
		*/
		void Problem::SolveLinearSystem()
		{
			/* 求解通用问题线性方程 */
			if (problemType_ == ProblemType::GENERIC_PROBLEM)
			{
				/* Hessian矩阵添加Lambda参数 */
				MatXX H = Hessian_;
				for (size_t i = 0; i < Hessian_.cols(); ++i) {
					H(i, i) += currentLambda_;
				}
				//delta_x_ = PCGSolver(H, b_, H.rows() * 2);
				delta_x_ = H.ldlt().solve(b_);
				if (non_linear_method_ == NonLinearMethod::Dog_Leg)
				{
					double beta = 0.;
					double alpha = b_.squaredNorm() / (b_.transpose()*Hessian_*b_);
					delta_x_sd_ = alpha*b_;
					delta_x_gn_ = delta_x_;
					if (delta_x_gn_.norm() >= dogleg_radius_)
					{
						if (delta_x_sd_.norm() >= dogleg_radius_)
						{
							delta_x_ = b_*(dogleg_radius_ / b_.norm());
						}
						else
						{
							const VecX& a = delta_x_sd_;
							const VecX& b = delta_x_gn_;
							double c = a.transpose()*(b - a);
							if (c <= 0)
							{
								beta = (-c + (std::sqrt)(c*c + (b - a).squaredNorm()*
									(dogleg_radius_*dogleg_radius_ - a.squaredNorm()))) /
									(b - a).squaredNorm();
							}
							else
							{
								beta = (dogleg_radius_*dogleg_radius_ - a.squaredNorm()) /
									(c + (std::sqrt)(c*c + (b - a).squaredNorm()*
									(dogleg_radius_*dogleg_radius_ - a.squaredNorm())));
							}
						}
						delta_x_ = delta_x_sd_ + beta*(delta_x_gn_ - delta_x_sd_);
					}
				}

				/* 求解SLAM问题线性方程：利用舒尔补加速SLAM问题的求解 */
			}
			else
			{
				//TicToc t_Hmminv;
				/* 将信息矩阵分解为位姿部分Hpp和路标点部分Hmm */
				int reserve_size = ordering_poses_;
				int marg_size = ordering_landmarks_;
				MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
				MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
				MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
				VecX bpp = b_.segment(0, reserve_size);
				VecX bmm = b_.segment(reserve_size, marg_size);
				/* 对路标点对应的信息矩阵Hmm部分求逆 */
				MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
				// TODO:: use openMP
				for (auto landmarkVertex : idx_landmark_vertices_) {
					int idx = landmarkVertex.second->OrderingId() - reserve_size;
					int size = landmarkVertex.second->LocalDimension();
					Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
				}
				/* 求解位姿部分信息矩阵及残差的舒尔补 */
				MatXX tempH = Hpm * Hmm_inv;
				H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
				b_pp_schur_ = bpp - tempH * bmm;
				/* 求解位姿变量 */
				VecX delta_x_pp(VecX::Zero(reserve_size));
				for (ulong i = 0; i < ordering_poses_; ++i) {
					H_pp_schur_(i, i) += currentLambda_;
				}
				// TicToc t_linearsolver;
				delta_x_pp = H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;    
				delta_x_.head(reserve_size) = delta_x_pp;
				// std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;
				/* 求解路标点变量 */
				VecX delta_x_ll(marg_size);
				delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
				delta_x_.tail(marg_size) = delta_x_ll;
				//std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;

				/* 将LM的结果当作初始值，使用Dog_Leg方法获取迭代步长 */
				if (non_linear_method_ == NonLinearMethod::Dog_Leg)
				{
					double beta = 0.;
					double alpha = b_.squaredNorm() / (b_.transpose()*Hessian_*b_);
					delta_x_sd_ = alpha*b_;
					delta_x_gn_ = delta_x_;
					if (delta_x_gn_.norm() >= dogleg_radius_)
					{
						if (delta_x_sd_.norm() >= dogleg_radius_)
						{
							delta_x_ = b_*(dogleg_radius_ / b_.norm());
						}
						else
						{
							const VecX& a = delta_x_sd_;
							const VecX& b = delta_x_gn_;
							double c = a.transpose()*(b - a);
							if (c <= 0)
							{
								beta = (-c + (std::sqrt)(c*c + (b - a).squaredNorm()*
									(dogleg_radius_*dogleg_radius_ - a.squaredNorm()))) /
									(b - a).squaredNorm();
							}
							else
							{
								beta = (dogleg_radius_*dogleg_radius_ - a.squaredNorm()) /
									(c + (std::sqrt)(c*c + (b - a).squaredNorm()*
									(dogleg_radius_*dogleg_radius_ - a.squaredNorm())));
							}
						}
						delta_x_ = delta_x_sd_ + beta*(delta_x_gn_ - delta_x_sd_);
					}
				}
			}
		}

		/*!
		*  @brief 一次迭代求解完毕后，将状态量更新
		*  @detail 更新状态量时，需要将状态量值备份；并且根据新的线性化点更新先验残差
		*/
		void Problem::UpdateStates()
		{
			/* 更新各个顶点的参数值，更新前，记录该参数值 */
			for (auto vertex : verticies_)
			{
				vertex.second->BackUpParameters();

				ulong idx = vertex.second->OrderingId();
				ulong dim = vertex.second->LocalDimension();
				VecX delta = delta_x_.segment(idx, dim);
				vertex.second->Plus(delta);
			}

			/* 根据新的状态量，更新先验残差 */
			if (err_prior_.rows() > 0)
			{
				/* 做个备份，下一次迭代效果不好再返回 */
				b_prior_backup_ = b_prior_;
				err_prior_backup_ = err_prior_;

				/* 状态量更新后，可以在新的线性化点重新泰勒展开，更新先验残差 */
				b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);
				err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);
			}
		}

		/*!
		*  @brief 一次迭代中，并没有使损失函数下降，此时需要将状态量调整
		*         到上一状态，更新Lambada，重新迭代
		*  @detail 需调整到上一迭代过程的变量包括顶点参数值以及先验残差项
		*/
		void Problem::RollbackStates()
		{
			/* 将顶点参数值调整到上一状态 */
			for (auto vertex : verticies_)
			{
				vertex.second->RollBackParameters();
			}

			/* 将先验残差调整到上一状态 */
			if (err_prior_.rows() > 0)
			{
				b_prior_ = b_prior_backup_;
				err_prior_ = err_prior_backup_;
			}
		}

		/*!
		*  @brief 根据Hessian矩阵值计算初始Lambda，并计算初始迭代状态残差
		*  @detail 计算初始状态损失函数值，此时使用了鲁棒核函数表示，并计算
		*          Hessian矩阵的最大对角元素，当作初始Lambda的计算基准
		*/
		void Problem::ComputeLambdaInitLM()
		{
			/* 设置初始的迭代控制参数 */
			ni_ = 2.;
			currentLambda_ = 0.0;
			currentChi_ = 0.0;
			dogleg_radius_ = 1.0;

			/* 计算初始的损失函数值 */
			for (auto edge : edges_)
			{
				currentChi_ += edge.second->RobustChi2();
			}
			if (err_prior_.rows() > 0)
			{
				currentChi_ += err_prior_.norm();
			}
			currentChi_ *= 0.5;

			/* 计算Hessian矩阵对角元素最大值 */
			double maxDiagonal = 0;
			ulong size = Hessian_.cols();
			assert(Hessian_.rows() == Hessian_.cols() && "Hessian Is Not Square");
			for (ulong i = 0; i < size; ++i)
			{
				maxDiagonal = (std::max)((std::fabs)(Hessian_(i, i)), maxDiagonal);
			}

			/* 计算初始的阻尼因子 */
			double tau = 1e-5;
			maxDiagonal = std::min(1e10, maxDiagonal);
			currentLambda_ = tau * maxDiagonal;
		}

		/*!
		*  @brief Hessian矩阵添加Lambda元素：此处不用
		*/
		void Problem::AddLambdatoHessianLM() {
			ulong size = Hessian_.cols();
			assert(Hessian_.rows() == Hessian_.cols() && "Hessian Is Not Square");
			for (ulong i = 0; i < size; ++i) {
				Hessian_(i, i) += currentLambda_;
			}
		}

		/*!
		*  @brief Hessian矩阵去除Lambda元素：此处不用
		*/
		void Problem::RemoveLambdaHessianLM() {
			ulong size = Hessian_.cols();
			assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
			for (ulong i = 0; i < size; ++i) {
				Hessian_(i, i) -= currentLambda_;
			}
		}

		/*!
		*  @brief 判断当前迭代是否是满足要求的迭代
		*  @detail 通过计算参数rho，判断当前迭代是否是好的迭代：损失函数下降，
		*          同时根据rho更新Lambda
		*/
		bool Problem::IsGoodStepInLM()
		{
			if (non_linear_method_ == NonLinearMethod::Levenberge_Marquardt)
			{
				double scale = 0;
				scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
				scale += 1e-6;

				double tmpChi = 0.0;
				for (auto edge : edges_)
				{
					edge.second->ComputeResidual();
					tmpChi += edge.second->RobustChi2();
				}
				if (err_prior_.size() > 0)
				{
					tmpChi += err_prior_.norm();
				}
				tmpChi *= 0.5;
				double rho = (currentChi_ - tmpChi) / scale;

				if (rho > 0 && isfinite(tmpChi))
				{
					double alpha = 1. - pow((2 * rho - 1), 3);
					alpha = std::min(alpha, 2. / 3.);
					double scaleFactor = (std::max)(1. / 3., alpha);
					currentLambda_ *= scaleFactor;
					ni_ = 2;
					currentChi_ = tmpChi;
					return true;
				}
				else
				{
					currentLambda_ *= ni_;
					ni_ *= 2;
					return false;
				}
			}
			else if (non_linear_method_ == NonLinearMethod::Dog_Leg)
			{
				double scale = 0.;
				scale = delta_x_.transpose()*b_;
				scale -= 0.5*delta_x_.transpose()*Hessian_*delta_x_;
				scale += 1e-6;

				double tmpChi = 0.;
				for (auto edge : edges_)
				{
					edge.second->ComputeResidual();
					tmpChi += edge.second->RobustChi2();
				}
				if (err_prior_.size() > 0)
				{
					tmpChi += err_prior_.squaredNorm();
				}
				tmpChi *= 0.5;

				double rho = (currentChi_ - tmpChi) / scale;
				if (rho > 0 && isfinite(tmpChi))
				{
					currentChi_ = tmpChi;
					if (rho > 0.75)
					{
						dogleg_radius_ = (std::max)(dogleg_radius_, 3.0*delta_x_.norm());
					}
					else if (rho < 0.25)
					{
						dogleg_radius_ /= 2.0;
					}
					return true;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}

		}

		/*!
		*  @brief 通过共轭梯度下降法计算Ax=b的解
		*/
		VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1)
		{
			assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
			int rows = b.rows();
			int n = maxIter < 0 ? rows : maxIter;
			VecX x(VecX::Zero(rows));
			MatXX M_inv = A.diagonal().asDiagonal().inverse();
			VecX r0(b);  // initial r = b - A*0 = b
			VecX z0 = M_inv * r0;
			VecX p(z0);
			VecX w = A * p;
			double r0z0 = r0.dot(z0);
			double alpha = r0z0 / p.dot(w);
			VecX r1 = r0 - alpha * w;
			int i = 0;
			double threshold = 1e-6 * r0.norm();
			while (r1.norm() > threshold && i < n)
			{
				i++;
				VecX z1 = M_inv * r1;
				double r1z1 = r1.dot(z1);
				double belta = r1z1 / r0z0;
				z0 = z1;
				r0z0 = r1z1;
				r0 = r1;
				p = belta * p + z1;
				w = A * p;
				alpha = r1z1 / p.dot(w);
				x += alpha * p;
				r1 -= alpha * w;
			}
			return x;
		}

		/*!
		*  @brief 边缘化掉所有与margVertexs相链接的edge：包括视觉edge与IMU edge
		*  @detail 若某个LandMark与Pose相连，但是又不想边缘化，那就把Edge去掉
		*          通过这个函数的内容可以发现这里边缘化始终维护了相机Pose相关的信息，
		*		   而对于路标点的先验信息，则没有维护，是否VINS中也是这个思路呢？？
		*  @param[in]	margVertex	当前优化问题中需要边缘化掉的状态量
		*  @param[in]	pose_dim	当前优化问题中所有帧的P、V、Q、Bg、Ba的维度
		*/
		bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex>> margVertexs, int pose_dim)
		{
			/* 设置优化问题中各类顶点的ID */
			SetOrdering();

			/* 获取待边缘化相机位姿对应的视觉edge以及IMU edge*/
			std::vector<std::shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

			/* 获取待边缘化相机姿态相连的路标点信息，并重新设定路标点序号 */
			int marg_landmark_size = 0;
			std::unordered_map<int, std::shared_ptr<Vertex>> margLandmark;
			for (size_t i = 0; i < marg_edges.size(); ++i) {
				auto verticies = marg_edges[i]->Verticies();
				for (auto iter : verticies) {
					if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
						iter->SetOrderingId(pose_dim + marg_landmark_size);
						margLandmark.insert(std::make_pair(iter->Id(), iter));
						marg_landmark_size += iter->LocalDimension();
					}
				}
			}

			/* 确定Hessian矩阵的维度，并初始化Hessian矩阵 */
			int cols = pose_dim + marg_landmark_size;
			MatXX H_marg(MatXX::Zero(cols, cols));
			VecX b_marg(VecX::Zero(cols));

			/* 遍历所有edge，构造Hessian矩阵以及b矩阵 */
			int edge_cnt = 0;
			for (auto edge : marg_edges)
			{
				edge->ComputeResidual();
				edge->ComputeJacobians();
				auto jacobians = edge->Jacobians();
				auto verticies = edge->Verticies();
				edge_cnt++;

				assert(jacobians.size() == verticies.size());
				for (size_t i = 0; i < verticies.size(); ++i)
				{
					auto v_i = verticies[i];
					auto jacobian_i = jacobians[i];
					ulong index_i = v_i->OrderingId();
					ulong dim_i = v_i->LocalDimension();

					/* 获取鲁棒核函数的相关信息 */
					double drho;
					MatXX robustInfo(edge->Information().rows(), edge->Information().cols());
					edge->RobustInfo(drho, robustInfo);

					for (size_t j = i; j < verticies.size(); ++j)
					{
						auto v_j = verticies[j];
						auto jacobian_j = jacobians[j];
						ulong index_j = v_j->OrderingId();
						ulong dim_j = v_j->LocalDimension();

						MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

						assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());

						H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
						if (j != i)
						{
							H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
						}
					}
					b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
				}
			}
			std::cout << "Edge Factor Cnt: " << edge_cnt << std::endl;

			/* 边缘化LandMark */
			int reserve_size = pose_dim;
			if (marg_landmark_size > 0)
			{
				int marg_size = marg_landmark_size;
				MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
				MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
				MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
				VecX bpp = b_marg.segment(0, reserve_size);
				VecX bmm = b_marg.segment(reserve_size, marg_size);

				// TODO USE OpenMP
				MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
				for (auto iter : margLandmark)
				{
					int idx = iter.second->OrderingId() - reserve_size;
					int size = iter.second->LocalDimension();
					Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
				}

				MatXX tempH = Hpm * Hmm_inv;
				MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
				bpp = bpp - tempH * bmm;
				H_marg = Hpp;
				b_marg = bpp;
			}

			/* 如果已经有先验，那么必须加上之前先验的影响 */
			VecX b_prior_before = b_prior_;
			if (H_prior_.rows() > 0)
			{
				H_marg += H_prior_;
				b_marg += b_prior_;
			}

			/* 边缘化Pose和SpeedBias */
			int marg_dim = 0;
			for (int k = margVertexs.size() - 1; k >= 0; --k)
			{
				int idx = margVertexs[k]->OrderingId();
				int dim = margVertexs[k]->LocalDimension();

				marg_dim += dim;
				/* 将row i移动到矩阵最下面 */
				Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);								// 获取Hmm的idx行
				Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);	// 获取Hmm中temp_rows下面所有行
				H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;						// 将temp_botRows移动到temp_rows的位置
				H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;									// 将最下面一行替换为temp_rows

				/* 将col i移动到矩阵最右边 */
				Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);								// 获取Hmm的idx列
				Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);// 获取Hmm中temp_cols右边所有行
				H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;						// 将temp_rightCols移动到temp_cols的位置
				H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;									// 将最右边一列替换为temp_cols

				/* 按照同样的道理移动b矩阵中的元素 */
				Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
				Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
				b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
				b_marg.segment(reserve_size - dim, dim) = temp_b;
			}

			double eps = 1e-8;
			int m2 = marg_dim;
			int n2 = reserve_size - marg_dim;
			Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
			Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
				(saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()*
				saes.eigenvectors().transpose();

			Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
			Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
			Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
			Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
			Eigen::VectorXd brr = b_marg.segment(0, n2);
			Eigen::MatrixXd tempB = Arm * Amm_inv;
			H_prior_ = Arr - tempB * Amr;
			b_prior_ = brr - tempB * bmm2;

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
			Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
			Eigen::VectorXd S_inv = Eigen::VectorXd(
				(saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

			Eigen::VectorXd S_sqrt = S.cwiseSqrt();
			Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
			Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
			err_prior_ = -Jt_prior_inv_ * b_prior_;

			MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
			H_prior_ = J.transpose() * J;
			MatXX tmp_h = MatXX((H_prior_.array().abs() > 1e-9).select(H_prior_.array(), 0));
			H_prior_ = tmp_h;

			/* 边缘化顶点后，在优化问题中需要删除对应顶点 */
			for (size_t k = 0; k < margVertexs.size(); ++k) {
				RemoveVertex(margVertexs[k]);
			}

			for (auto landmarkVertex : margLandmark) {
				RemoveVertex(landmarkVertex.second);
			}

			return true;
		}

		/*!
		*  @brief 测试边缘化写法是否正确
		*/
		void Problem::TestMarginalize()
		{
			int idx = 1;
			int dim = 1;
			int reserve_size = 3;
			double delta1 = 0.1*0.1;
			double delta2 = 0.2*0.2;
			double delta3 = 0.3*0.3;

			int cols = 3;
			MatXX H_marg(MatXX::Zero(cols, cols));
			H_marg << 1. / delta1, -1. / delta1, 0,
				-1. / delta1, 1. / delta1 + 1. / delta2 + 1. / delta3, -1. / delta3,
				0., -1. / delta3, 1. / delta3;
			std::cout << "TEST Marg: Before Marg" << std::endl;
			std::cout << H_marg << std::endl;

			Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
			Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);

			H_marg.block(idx, 0, dim, reserve_size) = temp_botRows;
			H_marg.block(reserve_size - dim, 0, reserve_size - idx - dim, reserve_size) = temp_rows;
			// 将 col i 移动矩阵最右边
			Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim); // 从0行，marg 中间那个变量 列开始索引；总变量维度，marg变量维度
			Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim); // 

			H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols; // 0行，marg 中间那个变量 列；总变量维度，总-marg
			H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols; // 从0行，总共变量的维度-marg变量的维度 列；总变量维度，marg变量的维度

			std::cout << "---------- TEST Marg: 将变量移动到右下角------------" << std::endl;
			std::cout << H_marg << std::endl;

			/// 开始 marg ： schur
			double eps = 1e-8;
			int m2 = dim;
			int n2 = reserve_size - dim;   // 剩余变量的维度
			Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
			Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
				(saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
				saes.eigenvectors().transpose();

			Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
			Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
			Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);

			Eigen::MatrixXd tempB = Arm * Amm_inv;
			Eigen::MatrixXd H_prior = Arr - tempB * Amr;

			std::cout << "---------- TEST Marg: after marg------------" << std::endl;
			std::cout << H_prior << std::endl;
		}

	}
}






