#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "utility/thread_pool.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(ThreadPool, AddTask)
{
	int value = 0;
	const int num_tasks = 100;
	{
		ThreadPool thread_pool(2);

		std::condition_variable condition;
		std::mutex mutex;

		for (int it = 0; it < num_tasks; ++it) {
			thread_pool.AddTask([&]() {
				std::lock_guard<std::mutex> lock(mutex);
				++value;
				condition.notify_all();
			});
		}

		std::unique_lock<std::mutex> lock(mutex);
		condition.wait(lock, [&]() {return value == num_tasks; });
	}

	EXPECT_EQ(num_tasks, value);
}