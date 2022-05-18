#pragma once

#include <memory>
#include <string>
#include "eigen_types.h"
#include <Eigen/Dense>
#include "loss_function.h"

namespace myslam {
namespace backend {

class Vertex;

/**
 * 边负责计算残差：残差=预测-观测，维度在构造函数中定义
 * 代价函数是=残差*信息*残差，是一个数值，由后端求和后最小化
 */
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/*!
	*  @brief 边构造函数：自动分配雅可比空间
	*  @param[in]	residual_dimension	残差维度
	*  @param[in]	num_verticies		当前边链接顶点数量
	*  @param[in]	verticies_types		当前边链接顶点类型
	*/
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string> &verticies_types = std::vector<std::string>());

	/*!
	*  @brief 边析构函数
	*/
    virtual ~Edge();

	/*!
	*  @brief 获得当前边的ID
	*  @return	unsigned long	当前边ID
	*/
    unsigned long Id() const { 
		return id_;
	}

	/*!
	*  @brief 为当前边添加对应顶点
	*  @param[in]	vertex	待添加的顶点
	*  @return		bool	顶点是否添加成功
	*/
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

	/*!
	*  @brief 为当前边设置一些顶点
	*  @param[in]	vertices	待设置的顶点
	*  @return		bool		顶点是否设置成功
	*/
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
        verticies_ = vertices;
        return true;
    }

	/*!
	*  @brief 返回当前边索引为i的顶点
	*  @param[in]	i						顶点索引
	*  @return		std::shared_ptr<Vertex>	顶点指针
	*/
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

	/*!
	*  @brief 返回当前边对应的所有顶点
	*  @return	std::vector<std::shared_ptr<Vertex>>	当前边对应顶点
	*/
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }

	/*!
	*  @brief 返回当前边对应顶点个数
	*  @return	size_t	当前边对应顶点个数
	*/
    size_t NumVertices() const {
		return verticies_.size(); 
	}

	/*!
	*  @brief 返回当前边类型信息：在子类中实现
	*  @return	std::string	当前边类型
	*/
    virtual std::string TypeInfo() const = 0;

	/*!
	*  @brief 计算当前边对应的残差：在子类中实现
	*/
    virtual void ComputeResidual() = 0;

	/*!
	*  @brief 计算当前边对应的雅可比：在子类中实现
	*/
    virtual void ComputeJacobians() = 0;

	/*!
	*  @brief 计算当前边对Hessian矩阵影响：在子类中实现
	*/
    //virtual void ComputeHessionFactor() = 0;

	/*!
	*  @brief 获取代价函数值：乘信息矩阵
	*  @return	double	代价函数值
	*/
    double Chi2() const;

	/*!
	*  @brief 获取添加鲁棒核函数的代价函数值：乘信息矩阵
	*  @return	double	添加鲁棒核函数的代价函数值
	*/
    double RobustChi2() const;

	/*!
	*  @brief 获取观测值残差：残差=预测-观测
	*  @return	VecX	观测值残差
	*/
    VecX Residual() const { 
		return residual_; 
	}

	/*!
	*  @brief 获取当前边对顶点的雅可比
	*  @return	std::vector<MatXX>	雅可比
	*/
    std::vector<MatXX> Jacobians() const { 
		return jacobians_; 
	}

	/*!
	*  @brief 设置当前边信息矩阵
	*  @param[in]	information	当前边信息矩阵
	*/
    void SetInformation(const MatXX &information) {
        information_ = information;
        sqrt_information_ = 
			Eigen::LLT<MatXX>(information_).matrixL().transpose();
    }

	/*!
	*  @brief 获取当前边信息矩阵
	*  @return	MatXX	当前边信息矩阵
	*/
    MatXX Information() const {
        return information_;
    }

	/*!
	*  @brief 获取当前边信息矩阵的一半
	*  @return	MatXX	当前边信息矩阵的一半
	*/
    MatXX SqrtInformation() const {
        return sqrt_information_;
    }

	/*!
	*  @brief 设置当前边鲁棒核函数
	*  @param[in]	ptr	当前边鲁棒核函数
	*/
    void SetLossFunction(LossFunction* ptr){
		lossfunction_ = ptr;
	}

	/*!
	*  @brief 获取当前边鲁棒核函数
	*  @return	LossFunction*	当前边鲁棒核函数
	*/
    LossFunction* GetLossFunction(){ 
		return lossfunction_;
	}

    void RobustInfo(double& drho, MatXX& info) const;

	/*!
	*  @brief 设置当前边的观测信息
	*  @param[in]	observation	当前边的观测信息
	*/
    void SetObservation(const VecX &observation) {
        observation_ = observation;
    }

	/*!
	*  @brief 获取当前边的观测信息
	*  @return	VecX	当前边的观测信息
	*/
    VecX Observation() const { 
		return observation_; 
	}

	/*!
	*  @brief 检查当前边的信息是否全部设置
	*  @return	bool	当前边信息是否全部设置的标志
	*/
    bool CheckValid();

	/*!
	*  @brief 获取当前边在Problem中的ID
	*  @return	int	当前边在Problem中排序后ID
	*/
    int OrderingId() const { 
		return ordering_id_; 
	}

	/*!
	*  @brief 设置当前边在Problem中的ID
	*  @param[in]	int	当前边在Problem中排序后ID
	*/
    void SetOrderingId(int id) { 
		ordering_id_ = id; 
	};

protected:
	/*!< @brief 当前边ID */
    unsigned long id_;
	/*!< @brief 当前边在Problem中的ID */
    int ordering_id_;
	/*!< @brief 当前边链接的所有顶点的类型 */
    std::vector<std::string> verticies_types_;
	/*!< @brief 当前边链接的所有顶点 */
    std::vector<std::shared_ptr<Vertex>> verticies_;
	/*!< @brief 当前边的残差信息=预测-观测 */
    VecX residual_;
	/*!< @brief 当前边对所有链接的顶点的雅可比 */
    std::vector<MatXX> jacobians_;
	/*!< @brief 当前边对应的信息矩阵 */
    MatXX information_;
	/*!< @brief 当前边对应的信息矩阵的一半 */
	MatXX sqrt_information_;
	/*!< @brief 当前边对应的观测信息 */
    VecX observation_;

	/*!< @brief 当前边对应的鲁棒核函数 */
    LossFunction *lossfunction_;
};

}
}