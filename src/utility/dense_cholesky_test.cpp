#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "utility/dense_cholesky.h"

TEST(CUDADenseCholesky, InvalidOptionOnCreate) {
	Options options;
	ContextImpl context;
	options.context = &context;
	auto cuda_dence_solver = CUDADenseCholesky::Create(options);
	EXPECT_EQ(cuda_dence_solver, nullptr);
}

