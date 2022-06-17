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

	// atomically push an element onto queue.
	// if a thread was waiting for an element, wake it up
	void Push(const T& value)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(value);
		work_pending_condition_.notify_one();
	}

	// atomically pop an element from the queue.
	// if an element is present,return true.
	// if the queue was empty, return false.
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
	// Pops an element from the queue.
	// if an element is present, return true.
	// if the queue was empty, return false.
	// not thread-safe. must acquire the lock before calling.
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
	// if true, signals that callers of wait block waiting to pop an element off the queue.
	bool wait_{ true };

	// the mutex controls read and write access to the queue_ and stop_ varaible.
	std::mutex mutex_;
	std::condition_variable work_pending_condition_;
};