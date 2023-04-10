#include <cmath>

#include "utility/thread_pool.h"

/*!
*  @brief 获取当前平台可用线程数量
*  @param[in]	requested_num_threads	当前算法申请线程数量
*  @return		int						当前平台能够给予的线程数量
*/
int GetNumAllowedThreads(int requested_num_threads)
{
	return (std::min)(requested_num_threads, ThreadPool::MaxNumThreadsAvailable());
}

/*!
*  @brief 获取当前平台最大可用线程数量
*  @return	int	当前平台最大可用线程数量
*/
int ThreadPool::MaxNumThreadsAvailable()
{
	// hardware_concurrency() can return 0 if the value is not well defined or not computable
	const int num_hardware_threads = std::thread::hardware_concurrency();
	return num_hardware_threads == 0 ? INT_MAX : num_hardware_threads;
}

/*!
*  @brief 线程池默认构造函数
*/
ThreadPool::ThreadPool() = default;

/*!
*  @brief 线程池构造函数
*  @param[in]	num_threads	用于初始化线程池的线程数量
*/
ThreadPool::ThreadPool(int num_threads)
{
	Resize(num_threads);
}

/*!
*  @brief 线程池析构函数
*  @param[in]	num_threads	用于初始化线程池的线程数量 
*/
ThreadPool::~ThreadPool()
{
	std::lock_guard<std::mutex> lock(thread_pool_mutex_);
	// signals the thread wokers to stop and wait for them to finish all 
	// schedual tasks.
	Stop();
	for (std::thread& thread : thread_pool_) {
		thread.join();
	}
}

/*!
*  @brief 重置线程池中的线程数量
*  @param[in]	num_threads	用于重置线程池的线程数量
*/
void ThreadPool::Resize(int num_threads)
{
	std::lock_guard<std::mutex> lock(thread_pool_mutex_);

	const int num_current_threads = thread_pool_.size();
	if (num_current_threads >= num_threads) {
		return;
	}

	const int create_num_threads =
		GetNumAllowedThreads(num_threads) - num_current_threads;

	for (int it = 0; it < create_num_threads; ++it) {
		thread_pool_.emplace_back(&ThreadPool::ThreadMainLoop, this);
	}
}

/*!
*  @brief 向线程池任务队列中添加任务
*  @param[in]	func	待添加到线程池中的任务队列
*/
void ThreadPool::AddTask(const std::function<void()>& func)
{
	task_queue_.Push(func);
}

/*!
*  @brief 获取线程池中的线程数量
*  @return	int	线程池中的线程数量
*/
int ThreadPool::Size()
{
	std::lock_guard<std::mutex> lock(thread_pool_mutex_);
	return thread_pool_.size();
}

void ThreadPool::ThreadMainLoop()
{
	std::function<void()> task;
	while (task_queue_.Wait(&task)) {
		task();
	}
}

void ThreadPool::Stop()
{
	task_queue_.stopWaiters();
}