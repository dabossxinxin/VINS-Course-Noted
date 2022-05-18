#pragma once

#include "eigen_types.h"

namespace myslam {
namespace backend {

/*!< @brief 全局参数：记录顶点ID */
extern unsigned long global_vertex_id;

 /**
 * 优化问题顶点：需要定义优化变量的维度
 */
class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/*!
	*  @brief 顶点构造函数
	*  @param[in]  num_dimension	顶点自身的维度
	*  @param[in]  local_dimension	顶点参与优化时的参数化维度
	*/
	explicit Vertex(int num_dimension, int local_dimension = -1);

	/*!
	*  @brief 顶点析构函数
	*/
    virtual ~Vertex();

	/*!
	*  @brief 获取变量自身的维度
	*  @return	int	变量自身维度
	*/
    int Dimension() const;

	/*!
	*  @brief 获取变量参与优化时的参数化维度
	*  @return	int	变量参与优化时的参数化维度
	*/
    int LocalDimension() const;

	/*!
	*  @brief 获取当前顶点的ID
	*  @return	unsigned long	当前顶点ID
	*/
    unsigned long Id() const { 
		return id_; 
	}

	/*!
	*  @brief 获取当前顶点参数值
	*  @return	VecX	当前顶点参数值
	*/
    VecX Parameters() const { 
		return parameters_; 
	}

	/*!
	*  @brief 获取当前顶点参数值的引用
	*  @return	VecX&	当前顶点参数值的引用
	*/
    VecX &Parameters() {
		return parameters_; 
	}

	/*!
	*  @brief 设置当前顶点的参数值：设定顶点初始值时使用
	*  @param[in]	params	顶点参数值初始值
	*/
    void SetParameters(const VecX &params) { 
		parameters_ = params; 
	}

	/*!
	*  @brief 备份当前顶点的参数值：用于迭代中去除不好的估计
	*/
    void BackUpParameters() { 
		parameters_backup_ = parameters_; 
	}

	/*!
	*  @brief 将上一步迭代中的顶点参数值回滚到当前迭代顶点参数
	*/
    void RollBackParameters() {
		parameters_ = parameters_backup_; 
	}

	/*!
	*  @brief 顶点参数值加法：默认为向量加法，通过继承重定义
	*  @param[in]	delta	顶点参数值变化量
	*/
    virtual void Plus(const VecX &delta);

	/*!
	*  @brief 返回顶点的名称：在子类中实现
	*  @return	std::string	顶点类型名称
	*/
    virtual std::string TypeInfo() const = 0;

	/*!
	*  @brief 返回排序后顶点ID
	*  @return	int	排序后顶点ID
	*/
    int OrderingId() const { 
		return ordering_id_; 
	}

	/*!
	*  @brief 设置当前顶点排序后ID
	*  @param[in]	id	排序后顶点ID
	*/
    void SetOrderingId(unsigned long id) { 
		ordering_id_ = id; 
	}

	/*!
	*  @brief 固定当前顶点的参数值
	*  @param[in]	fixed	是否固定当前顶点参数值
	*/
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

	/*!
	*  @brief 获取当前顶点的参数是否被固定
	*  @return	bool	当前顶点参数值是否被固定
	*/
    bool IsFixed() const { 
		return fixed_; 
	}

protected:
	/*!< @brief 当前顶点参数值 */
    VecX			parameters_;
	/*!< @brief 当前顶点参数值的备份 */
    VecX			parameters_backup_;
	/*!< @brief 当前顶点优化时的参数化维度 */
    int				local_dimension_;
	/*!< @brief 当前顶点ID：自动生成 */
    unsigned long	id_;

	/*!< @brief 当前顶点排序后ID：
	在Problem中排序，用于寻找对应的雅可比块 */
    unsigned long ordering_id_ = 0;
	/*!< @brief 当前顶点是否固定的标志 */
    bool fixed_ = false;
};

}
}