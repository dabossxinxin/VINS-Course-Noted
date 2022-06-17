#include <cmath>

#include "utility/thread_pool.h"

// constrain the total number of threads to the amount the hardware can support.
int GetNumAllowedThreads(int requested_num_threads)
{
	return (std::min)(requested_num_threads, ThreadPool::MaxNumThreadsAvailable());
}

int ThreadPool::MaxNumThreadsAvailable()
{
	// hardware_concurrency() can return 0 if the value is not well defined or not computable
	const int num_hardware_threads = std::thread::hardware_concurrency();
	return num_hardware_threads == 0 ? INT_MAX : num_hardware_threads;
}

ThreadPool::ThreadPool() = default;

ThreadPool::ThreadPool(int num_threads)
{
	Resize(num_threads);
}

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

void ThreadPool::AddTask(const std::function<void()>& func)
{
	task_queue_.Push(func);
}

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