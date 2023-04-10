#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "glog/logging.h"

template <typename T>
class ConcurrentQueue
{
public:
	ConcurrentQueue() = default;

	/*!
	*  @brief 使用原子操作将元素放入队列中
	*  @detail 将元素放入并发队列后，通知线程处理
	*  @param[in]	value	待插入队列的元素
	*/
	void Push(const T& value)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(value);
		work_pending_condition_.notify_one();
	}

	/*!
	*  @brief 使用原子操作将队列头部元素删除
	*  @detail 若并发队列为空，则返回false
	*  @param[in]	value	待插入队列的元素
	*  @return		bool	是否成功删除队列元素的标志
	*/
	bool Pop(T* value)
	{
		CHECK(value != nullptr);

		std::lock_guard<std::mutex> lock(mutex_);
		return PopUnlocked(value);
	}

	// atomically pop an element from the queue.
	// block until one is available or stopwaiters is called.
	// return true if an element was successfully popped from the queue,
	// otherwise returns false.
	bool Wait(T* value)
	{
		CHECK(value != nullptr);

		std::unique_lock<std::mutex> lock(mutex_);
		/* 当任务为空且wait为true时，阻塞线程 */
		work_pending_condition_.wait(lock,
			[&]() {
			return !(wait_ && queue_.empty());
		});

		return PopUnlocked(value);
	}

	// Unlock all threads waiting to pop a value from the queue, and 
	// they will exit Wait() without getting a value. All future Wait 
	// requests will return immediately if no element is present until 
	// EnableWaiters is called.
	void stopWaiters()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		wait_ = false;
		work_pending_condition_.notify_all();
	}

	// Enable threads to block on wait calls.
	void EnableWaiters()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		wait_ = true;
	}

private:
	/*!
	*  @brief 将队列头部元素删除
	*  @detail 若并发队列为空，则返回false
	*          若并发队列不为空，则返回true
	*  @param[in/out]	value	队列中待删除元素
	*  @return			bool	是否成功删除队列元素的标志
	*/
	bool PopUnlocked(T* value)
	{
		if (queue_.empty())
		{
			return false;
		}

		*value = queue_.front();
		queue_.pop();

		return true;
	}

	std::queue<T> queue_;
	bool wait_{ true };
	std::mutex mutex_;
	std::condition_variable work_pending_condition_;
};