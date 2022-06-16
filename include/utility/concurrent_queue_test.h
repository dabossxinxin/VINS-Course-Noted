#include "utility/concurrent_queue.h"

//#include <gtest/gmock.h>
#include <gtest/gtest.h>

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