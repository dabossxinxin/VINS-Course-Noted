#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"

#ifdef USE_OPENMP
    #include <omp.h>
#endif

using namespace std;

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
void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

/*!
*  @brief 优化问题构造函数
*  @param[in]   problemType 优化问题类型
*/
Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

/*!
*  @brief 优化问题析构函数
*  @param[in]   problemType 优化问题类型
*/
Problem::~Problem() {
    std::cout << "Problem Is Deleted"<<std::endl;
    global_vertex_id = 0;
}

/*!
*  @brief 向优化问题中添加待待优化量[节点]
*  @param[in]	vertex	待优化顶点
*  @retuan		bool	是否成功添加顶点
*/
bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    /* 判断节点vertex是否已经添加到优化问题中 */
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        std::cout << "Vertex " << vertex->Id() << " Has Been Added Before" << std::endl;
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
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
void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    /* SLAM问题中添加Pose顶点 */
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    /* SLAM问题中添加LandMark顶点 */
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

/*!
*  @brief 添加顶点后，需要调整先验Hessian&先验残差的大小
*  @param[in]   v   新加入的顶点
*/
void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {
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
*  @param[in]   dim   需扩充的维度
*/
void Problem::ExtendHessiansPriorSize(int dim) {
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
bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose") ||
            type == string("VertexSpeedBias");
}

/*!
*  @brief 判断顶点是否为Pose顶点
*  @param[in]   v       输入顶点
*  @return      bool    输入节点是否为Pose节点
*/
bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

/*!
*  @brief 向优化问题中添加边
*  @param[in]	vertex	待添加边
*  @return		bool	是否成功添加边
*/
bool Problem::AddEdge(shared_ptr<Edge> edge) {
    /* 判断边edge是否已经添加到优化问题中 */
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        std::cerr << "Edge " << edge->Id() << " Has Been Added Before!";
        return false;
    }
    /* 更新vertexToEdge_，便于由vertex查询edge */
    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

/*!
*  @brief 获取指定顶点相链接的边
*  @param[in]   vertex                          需获取链接边的顶点
*  @return  std::vector<std::shared_ptr<Edge>>  输入顶点链接的边
*/
std::vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    std::vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {
        /* 在优化问题的所有边中寻找vertex节点对应的边 */
        if (edges_.find(iter->second->Id()) == edges_.end()) {
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
bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    /* 查询当前待删除的节点是否存在于优化问题的所有节点中 */
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        std::cout << "The Vertex " << vertex->Id() << " Is Not In The Problem!" << std::endl;
        return false;
    }
    /* 删除节点vertex对应的edge */
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }
    /* 更新数据结构idx_pose_verticies和idx_landmark_vertices */
    if (IsPoseVertex(vertex)) {
        idx_pose_vertices_.erase(vertex->Id());
    } else {
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
bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
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
bool Problem::Solve(int iterations) {
    /* 检查优化问题中是否具有顶点元素以及边元素 */
    if (edges_.size() == 0 || verticies_.size() == 0) {
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
    double last_chi_ = 1e20;
    while (!stop && (iter < iterations)) {
        /* 打印优化的关键信息 */
        std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        /* 不断尝试Lambda：直到成功迭代一步 */
        while (!oneStepSuccess && false_cnt < 10) {
            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            SolveLinearSystem();
            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
//            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
            // TODO:: 退出条件还是有问题, 好多次误差都没变化了，还在迭代计算，应该搞一个误差不变了就中止
//            if ( false_cnt > 10)
//            {
//                stop = true;
//                break;
//            }
            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
//                std::cout << " get one step success\n";

                // 在新线性化点 构建 hessian
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        // TODO:: 应该改成前后两次的误差已经不再变化
//        if (sqrt(currentChi_) <= stopThresholdLM_)
//        if (sqrt(currentChi_) < 1e-15)
        if(last_chi_ - currentChi_ < 1e-6)
        {
            //std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
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
*  @brief 统计优化问题的维度：vertex维度及edge维度
*/
void Problem::SetOrdering() {
    /* 初始化维度 */
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;
    /* 统计优化问题中顶点维度 */
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();
        /* 如果是SLAM问题：需要分别统计Pose与LandMark维度 */
        if (problemType_ == ProblemType::SLAM_PROBLEM) {
            AddOrderingSLAM(vertex.second);
        }
    }
    /* 此处需要把LandMark的Ordering加上Pose的数量：保证LandMark顺序在后Pose在前 */
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }
    //CHECK_EQ(CheckOrdering(), true);
}

/*!
*  @brief 检查Ordering是否正确
*/
bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        /* 检查Pose节点的顺序是否正确 */
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
        /* 检查LandMark节点的顺序是否正确 */
        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

/*!
*  @brief 构造大Hessian矩阵
*/
void Problem::MakeHessian() {
    /* 记录Hessian矩阵的构造时间 */
    TicToc t_h;
    /* 直接构造Hessian矩阵 */
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));
	/* 遍历edge_：计算Hessian矩阵 */
	#ifdef USE_OPENMP
//#pragma omp parallel for
	#endif
    for (auto &edge: edges_) {

        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            /* 为当前边添加鲁棒核函数 */
			double drho = 1.0;
            MatXX robustInfo(edge.second->Information().rows(),edge.second->Information().cols());
            edge.second->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
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

    if(H_prior_.rows() > 0)
    {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex: verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

/*!
*  @brief 求解线性方程
*/
void Problem::SolveLinearSystem() {
    /* 求解通用问题线性方程 */
    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        /* Hessian矩阵添加Lambda参数 */
        MatXX H = Hessian_;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        //delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);
    /* 求解SLAM问题线性方程：利用舒尔补加速SLAM问题的求解 */
    } else {
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
        delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;    
        delta_x_.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;
        /* 求解路标点变量 */
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        delta_x_.tail(marg_size) = delta_x_ll;
        //std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }
}

void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }
}

void Problem::RollbackStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    // Roll back prior_
    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

/// LM
void Problem::ComputeLambdaInitLM() {
	/* 设置初始的迭代控制参数 */
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

	/* 计算初始的损失函数值 */
    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
	if (err_prior_.rows() > 0) {
		currentChi_ += err_prior_.norm();
	}
    currentChi_ *= 0.5;
	
    //stopThresholdLM_ = 1e-10 * currentChi_;
	/* 计算Hessian矩阵对角元素最大值 */
    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian Is Not Square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = (std::max)((std::fabs)(Hessian_(i, i)), maxDiagonal);
    }
	/* 计算初始的阻尼因子 */
	double tau = 1e-5;
    maxDiagonal = std::min(1e+10, maxDiagonal);
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

bool Problem::IsGoodStepInLM() {
    double scale = 0;
//    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
//    scale += 1e-3;    // make sure it's non-zero :)
    scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-6;    // make sure it's non-zero :)

    // recompute residuals after update state
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();
    tempChi *= 0.5;          // 1/2 * err^2

    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        return false;
    }
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
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
    while (r1.norm() > threshold && i < n) {
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

/*
 *  marg 所有和 frame 相连的 edge: imu factor, projection factor
 *  如果某个landmark和该frame相连，但是又不想加入marg, 那就把改edge先去掉
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int pose_dim) {

    SetOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // 构建 Hessian 的时候 pose 的顺序不变，landmark的顺序要重新设定
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto verticies = marg_edges[i]->Verticies();
        for (auto iter : verticies) {
            if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                iter->SetOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    int cols = pose_dim + marg_landmark_size;
    /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
    for (auto edge: marg_edges) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
                // 所有的信息矩阵叠加起来
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // 对称的下三角
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
        }

    }
        std::cout << "edge factor cnt: " << ii <<std::endl;

    /// marg landmark
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto iter: margLandmark) {
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

    VecX b_prior_before = b_prior_;
    if(H_prior_.rows() > 0)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // index 大的先移动
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {

        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
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
    MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        RemoveVertex(margVertexs[k]);
    }

    for (auto landmarkVertex: margLandmark) {
        RemoveVertex(landmarkVertex.second);
    }

    return true;

}

}
}






