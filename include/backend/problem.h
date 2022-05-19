#pragma once

#include <unordered_map>
#include <map>
#include <memory>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
//typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>>                HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>>        HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>>   HashVertexIdToEdge;

class Problem {
public:
    /*! * @brief 保证向量空间内存对齐 */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    /**
     * 问题类型：SLAM问题/通用问题
     * SLAM问题：1、Pose和Landmark是区分开的，Hessian以稀疏方式存储
     *           2、SLAM问题只接受一些特定的Vertex和Edge
     * 通用问题：hessian是稠密矩阵，不做特殊处理
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    /*!
	*  @brief 优化问题构造函数
	*  @param[in]  problemType	优化问题类型
	*/
    Problem(ProblemType problemType);

    /*!
	*  @brief 优化问题析构函数
	*/
    ~Problem();

    /*!
	*  @brief 向优化问题中添加待待优化量
	*  @param[in]	vertex	待优化顶点
	*  @return		bool	是否成功添加顶点
	*/
    bool AddVertex(std::shared_ptr<Vertex> vertex);

    /*!
	*  @brief 移除优化问题中的优化变量
	*  @param[in]	vertex	待优化顶点
	*  @return		bool	是否成功移除顶点
	*/
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    /*!
	*  @brief 向优化问题中添加边
	*  @param[in]	vertex	待添加边
	*  @return		bool	是否成功添加边
	*/
    bool AddEdge(std::shared_ptr<Edge> edge);

    /*!
	*  @brief 移除优化问题中的边
	*  @param[in]	vertex	待移除边
	*  @return		bool	是否成功移除边
	*/
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /*!
	*  @brief 获取优化问题中被判定为外点的边：方便前端的操作
	*  @param[in]	outlier_edges	外点边
	*/
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

    /*!
	*  @brief 求解当前的优化问题
	*  @param[in]	iterations  非线性优化迭代次数
    *  @return      bool        非线性优化问题是否成功收敛
	*/
    bool Solve(int iterations = 10);

    /*!
	*  @brief SLAM问题中边缘化指定Frame&LandMark
	*  @param[in]	frameVertex         待边缘化的Frame
    *  @param[in]   landmarkVerticies   待边缘化的LandMark
    *  @return      bool                是否成功边缘化
	*/
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);

    /*!
	*  @brief SLAM问题中边缘化指定Frame
	*  @param[in]	frameVertex         待边缘化的Frame
    *  @return      bool                是否成功边缘化
	*/
    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

    /*!
	*  @brief SLAM问题中边缘化指定Frame
	*  @param[in]	frameVertex         待边缘化的Frame
    *  @param[in]	pose_dim            Frame的维度
    *  @return      bool                是否成功边缘化
	*/
    bool Marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex,int pose_dim);

	void TestMarginalize();

    MatXX GetHessianPrior(){ 
        return H_prior_;
    }

    VecX GetbPrior(){ 
        return b_prior_;
    }

    VecX GetErrPrior(){
         return err_prior_;
    }

    MatXX GetJtPrior(){
         return Jt_prior_inv_;
    }

    void SetHessianPrior(const MatXX& H){
        H_prior_ = H;
    }

    void SetbPrior(const VecX& b){
        b_prior_ = b;
    }

    void SetErrPrior(const VecX& b){
        err_prior_ = b;
    }

    void SetJtPrior(const MatXX& J){
        Jt_prior_inv_ = J;
    }

    void ExtendHessiansPriorSize(int dim);

    void TestComputePrior();

private:

    /*!
	*  @brief Solve的实现：求解通用非线性优化问题
	*  @param[in]	iterations  非线性优化迭代次数
    *  @return      bool        非线性优化问题是否成功收敛
	*/
    bool SolveGenericProblem(int iterations);

    /*!
	*  @brief Solve的实现：求解SLAM非线性优化问题
	*  @param[in]	iterations  非线性优化迭代次数
    *  @return      bool        非线性优化问题是否成功收敛
	*/
    bool SolveSLAMProblem(int iterations);

    /*!
	*  @brief 统计优化问题的维度：vertex维度及edge维度
	*/
    void SetOrdering();

    /*!
	*  @brief SLAM问题中设置新加入顶点的顺序
    *  @param[in]   v   新加入SLAM问题中的顶点
	*/
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /*!
	*  @brief 构造大Hessian矩阵
	*/
    void MakeHessian();

    /*!
	*  @brief Schur求解SBA问题
	*/
    void SchurSBA();

    /*!
	*  @brief 求解线性方程
	*/
    void SolveLinearSystem();

    /*!
	*  @brief 更新状态变量
	*/
    void UpdateStates();

    /*!
	*  @brief Update后残差变大时，回退回去重新计算
	*/
    void RollbackStates();

    /*!
	*  @brief 计算并更新Prior部分
	*/
    void ComputePrior();

    /*!
	*  @brief 判断顶点是否为Pose顶点
    *  @param[in]   v       输入顶点
    *  @return      bool    输入节点是否为Pose节点
	*/
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /*!
	*  @brief 判断顶点是否为LandMark顶点
    *  @param[in]   v       输入顶点
    *  @return      bool    输入节点是否为LandMark节点
	*/
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /*!
	*  @brief 添加顶点后，需要调整先验Hessian&先验残差的大小
    *  @param[in]   v   新加入的顶点
	*/
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /*!
	*  @brief 检查Ordering是否正确
	*/
    bool CheckOrdering();

    void LogoutVectorSize();

    /*!
	*  @brief 获取指定顶点相链接的边
    *  @param[in]   vertex                          需获取链接边的顶点
    *  @return  std::vector<std::shared_ptr<Edge>>  输入顶点链接的边
	*/
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /*!
	*  @brief 计算LM方法的初始Lambda
	*/
    void ComputeLambdaInitLM();

    /*!
	*  @brief Hessian矩阵添加Lambda元素
	*/
    void AddLambdatoHessianLM();

    /*!
	*  @brief Hessian矩阵去除Lambda元素
	*/
    void RemoveLambdaHessianLM();

    /*!
	*  @brief 判断上一步迭代是否OK，从而确定Lambda的取值
	*/
    bool IsGoodStepInLM();

    /*!
	*  @brief PCG方法求解线性方程
    *  @param[in]   A       线性方程信息矩阵
    *  @param[in]   b       线性方程残差矩阵
    *  @param[in]   maxIter 最大迭代次数
    *  @return      VecX    线性方程求解结果
	*/
    VecX PCGSolver(const MatXX &A, const VecX &b, int maxIter);

private:
    /*!< @brief 当前Lambda */
    double currentLambda_;
    /*!< @brief 当前残差平方和 */
    double currentChi_;
    /*!< @brief LM迭代退出阈值条件 */
    double stopThresholdLM_;    
    /*!< @brief 控制Lambda缩放大小 */
    double ni_;                 

    /*!< @brief 非线性优化问题类型 */
    ProblemType problemType_;

    /*!< @brief 优化问题中Jt*J */
    MatXX Hessian_;
    /*!< @brief 优化问题中-Jt*f */
    VecX b_;
    /*!< @brief 优化问题步长 */
    VecX delta_x_;

    /*!< @brief 先验Hessian矩阵 */
    MatXX H_prior_;
    /*!< @brief 先验残差矩阵 */
    VecX b_prior_;
    /*!< @brief 上一迭代过程先验残差矩阵：用于回退 */
    VecX b_prior_backup_;
    /*!< @brief TODO：上一迭代过程先验误差：用于回退 */
    VecX err_prior_backup_;
    /*!< @brief 先验雅可比的逆 */
    MatXX Jt_prior_inv_;
    /*!< @brief TODO：先验误差 */
    VecX err_prior_;

    /*!< @brief SBA的Pose部分 */
    MatXX H_pp_schur_;
    /*!< @brief SBA的Pose部分 */
    VecX b_pp_schur_;
    /*!< @brief JtJ的Pose部分 */
    MatXX H_pp_;
    /*!< @brief -Jtf的Pose部分 */
    VecX b_pp_;
    /*!< @brief JtJ的LandMark部分 */
    MatXX H_ll_;
    /*!< @brief -Jtf的LandMark部分 */
    VecX b_ll_;

    /*!< @brief 当前优化问题中的所有顶点 */
    HashVertex verticies_;
    /*!< @brief 当前优化问题中的所有边 */
    HashEdge edges_;
    /*!< @brief 当前优化问题中由Vertex ID查询对应的边 */
    HashVertexIdToEdge vertexToEdge_;

    /*!< @brief Pose Order相关 */
    ulong ordering_poses_ = 0;
    /*!< @brief LandMark Order相关 */
    ulong ordering_landmarks_ = 0;
    /*!< @brief 通用问题顶点Order相关 */
    ulong ordering_generic_ = 0;
    /*!< @brief Ordering后的Pose顶点 */
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;
    /*!< @brief Ordering后的LandMark顶点 */
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;
    
    /*!< @brief 需边缘化的顶点<Ordering_id_, Vertex> */
    HashVertex verticies_marg_;

    /*!< @brief 是否开启调试模式的标志 */
    bool bDebug = false;
    /*!< @brief 构造Hessian矩阵的时间 */
    double t_hessian_cost_ = 0.0;
    /*!< @brief PCG线性方程求解的时间 */
    double t_PCGsovle_cost_ = 0.0;
};

}
}
