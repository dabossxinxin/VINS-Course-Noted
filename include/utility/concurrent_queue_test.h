#include "utility/concurrent_queue.h"

//#include <gtest/gmock.h>
#include <gtest/gtest.h>

/*!
*  @brief 测试并发队列插入删除操作，单个线程操作
*  @detail 测试并发队列能否正常删除元素
*          测试并发队列能否按照正确顺序删除元素
*/
TEST(ConcurrentQueue, PushPop)
{
	ConcurrentQueue<int> queue;

	const int num_to_add = 10;
	for (int i = 0; i < num_to_add; ++i) {
		queue.Push(i);
	}

	for (int i = 0; i < num_to_add; ++i) {
		int value;
		ASSERT_TRUE(queue.Pop(&value));
		EXPECT_EQ(i, value);
	}
}

/*!
*  @brief 测试并发队列在StopWaiter后插入删除操作，单个线程操作
*  @detail 测试并发队列能否正常删除元素
*          测试并发队列能否按照正确顺序删除元素
*/
TEST(ConcurrentQueue, PushPopAfterStopWaiters)
{
	ConcurrentQueue<int> queue;

	const int num_to_add = 10;
	int value;

	ASSERT_FALSE(queue.Pop(&value));

	for (int i = 0; i < num_to_add; ++i) {
		queue.Push(i);
	}

	queue.stopWaiters();

	for (int i = 0; i < num_to_add; ++i) {
		ASSERT_TRUE(queue.Pop(&value));
		EXPECT_EQ(i, value);
	}

	ASSERT_FALSE(queue.Pop(&value));

	const int offset = 123;
	for (int i = 0; i < num_to_add; ++i) {
		queue.Push(i + offset);
	}

	for (int i = 0; i < num_to_add; ++i) {
		int value;
		ASSERT_TRUE(queue.Pop(&value));
		EXPECT_EQ(i + offset, value);
	}

	ASSERT_FALSE(queue.Pop(&value));

	queue.stopWaiters();

	queue.Push(13456);
	ASSERT_TRUE(queue.Pop(&value));
	EXPECT_EQ(13456, value);
}

TEST(ConcurrentQueue, EnsureWaitBlocks)
{
	ConcurrentQueue<int> queue;

	int value = 0;
	bool valid_value = false;
	bool waiting = false;
	std::mutex mutex;

	std::thread thread([&]() {
		{
			std::lock_guard<std::mutex> lock(mutex);
			waiting = true;
		}

		int element = 87987;
		bool valid = queue.Wait(&element);

		{
			std::lock_guard<std::mutex> lock(mutex);
			waiting = false;
			value = element;
			valid_value = valid;
		}
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	{
		std::lock_guard<std::mutex> lock(mutex);
		EXPECT_TRUE(waiting);
		ASSERT_FALSE(valid_value);
		ASSERT_EQ(0, value);
	}

	queue.Push(13456);

	thread.join();

	EXPECT_TRUE(valid_value);
	EXPECT_EQ(13456, value);
}