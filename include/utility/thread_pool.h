#pragma once

#include <mutex>
#include <thread>
#include <vector>

#include "utility/concurrent_queue.h"

class ThreadPool
{
public:
	// returns the maximum number of hardware threads.
	static int MaxNumThreadsAvailable();

	// 
	ThreadPool();

	// instantiates a thread pool with min(MaxNumThreadsAvailable,num_threads)
	// number of threads
	explicit ThreadPool(int num_threads);

	// signals the workers to stop and waits for them to finish any tasks that
	// have been scheduled.
	~ThreadPool();

	// resize the thread pool if it is currently less than the requested number
	// of threads,the thread pool will be resized to min(MaxNumThreadsAvailable,
	// num_threads) number of threads.resize does not support reducing the pool
	// size.if a smaller number of threads is requested,the thread pool remains 
	// the same size.It is safe to resize the thread pool while the workers are 
	// executing tasks,and the resizing is guaranteed to complete upon return.
	void Resize(int num_threads);

	// Adds a task to the queue and wakes up a blocked thread.if the thread pool
	// size is greater than zero,then the task will be executed by a currently
	// thread.if the thread pool has no threads, then the task will never be 
	// executed and the user should use Resize() to create a non-empty thread pool
	void AddTask(const std::function<void()>& func);

	// returns the current size of the thread pool
	int Size();
private:
	// Main loop for the threads whitch blocks on the task queue until work becomes
	// available. it will return if and only if stop has been called.
	void ThreadMainLoop();

	// signals all the threads to stop.it does not block until the threads are finished.
	void Stop();
	
	/*!< @brief 保存线程池中需要处理的所有任务队列 */
	ConcurrentQueue<std::function<void()>> task_queue_;
	/*!< @brief 保存线程池中所有的用于处理任务的线程 */
	std::vector<std::thread> thread_pool_;
	std::mutex thread_pool_mutex_;
};