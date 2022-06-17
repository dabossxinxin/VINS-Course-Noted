#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "glog/logging.h"
#include "gtest/gtest.h"

//#include "utility/dense_cholesky.h"
#include "cuda_runtime.h"
//#include "cusolverDn.h"
#include "cublas.h"
//#include "utility/concurrent_queue_test.h"
#include "utility/thread_pool_test.h"

//TEST(CUDADenseCholesky, InvalidOptionOnCreate) {
//	Options options;
//	ContextImpl context;
//	options.context = &context;
//	auto cuda_dence_solver = CUDADenseCholesky::Create(options);
//	EXPECT_NE(cuda_dence_solver, nullptr);
//}

//TEST(CUDADenseCholesky, Cholesky4x4Matrix) {
//	Eigen::Matrix4d A;
//	A << 4, 12, -16, 0,
//		12, 37, -43, 0,
//		-16, -43, 98, 0,
//		0, 0, 0, 1;
//
//	Options options;
//	const Eigen::Vector4d b = Eigen::Vector4d::Ones();
//	ContextImpl context;
//	options.context = &context;
//	auto dense_cuda_solver = CUDADenseCholesky::Create(options);
//	ASSERT_NE(dense_cuda_solver, nullptr);
//	std::string error_string;
//	ASSERT_EQ(dense_cuda_solver->Factorize(A.cols(), A.data(), &error_string),
//		LinearSolverTerminationType::SUCCESS);
//	Eigen::Vector4d x = Eigen::Vector4d::Zero();
//	ASSERT_EQ(dense_cuda_solver->Solve(b.data(), x.data(), &error_string),
//		LinearSolverTerminationType::SUCCESS);
//	EXPECT_NEAR(x(0), 113.75 / 3.0, std::numeric_limits<double>::epsilon() * 10);
//	EXPECT_NEAR(x(1), -31.0 / 3.0, std::numeric_limits<double>::epsilon() * 10);
//	EXPECT_NEAR(x(2), 5.0 / 3.0, std::numeric_limits<double>::epsilon() * 10);
//	EXPECT_NEAR(x(3), 1.0000, std::numeric_limits<double>::epsilon() * 10);
//}

#define N 200

/* 矩阵相乘 */
void simple_sgemm(const float* A, const float* B, float* C) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			float s = 0;
			for (int k = 0; k < N; ++k) {
				s += A[k*N + i] * B[j*N + k];
			}
			C[j*N + i] = s;
		}
	}
}

void simple_sgemm_eigen(const float* A, const float* B, float* C) {
	Eigen::MatrixXf A_(N, N);
	Eigen::MatrixXf B_(N, N);
	Eigen::MatrixXf C_(N, N);
	for (int i = 0; i < N*N; ++i) {
		A_(i) = A[i];
		B_(i) = B[i];
	}
	clock_t t_cpu_start = clock();
	C_ = A_*B_;
	C = C_.data();
	clock_t t_cpu_end = clock();
	float cpu_t = t_cpu_end - t_cpu_start;
	printf("N=%4d, CPU=%.6fms(%.3fGflops)\n",
		N, cpu_t, 1e-9*N*N*N * 2 / cpu_t);
}

void printMatrix(int m, int n, const double* A, int lda, const char* name) {
	for (int row = 0; row < m; ++row) {
		for (int col = 0; col < n; ++col) {
			double Areg = A[row + col*lda];
			printf("%s(%d,%d)=%f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int main(int argc, char* argv[]) {
	testing::InitGoogleTest(&argc, argv);
	int node = RUN_ALL_TESTS();

	//float *h_A = (float*)malloc(N*N * sizeof(float));
	//float *h_B = (float*)malloc(N*N * sizeof(float));
	//float *h_C = (float*)malloc(N*N * sizeof(float));
	//float *h_C_ref = (float*)malloc(N*N * sizeof(float));
	//float *d_A, *d_B, *d_C;

	//for (int i = 0; i < N*N; i++) {
	//	h_A[i] = rand() / (float)RAND_MAX;
	//	h_B[i] = rand() / (float)RAND_MAX;
	//}

	//printf("SimpleCuBals Test Running..\n");

	//clock_t t_gpu_start = clock();
	//cublasInit();
	//cublasAlloc(N*N, sizeof(float), (void**)&d_A);
	//cublasAlloc(N*N, sizeof(float), (void**)&d_B);
	//cublasAlloc(N*N, sizeof(float), (void**)&d_C);
	//cublasSetVector(N*N, sizeof(float), h_A, 1, d_A, 1);
	//cublasSetVector(N*N, sizeof(float), h_B, 1, d_B, 1);

	//cudaThreadSynchronize();
	//cublasSgemm('n', 'n', N, N, N, 1.0f, d_A, N, d_B, N, 0.0f, d_C, N);
	//cudaThreadSynchronize();
	//cublasGetVector(N*N, sizeof(float), d_C, 1, h_C, 1);
	//clock_t t_gpu_end = clock();
	//float gpu_t = t_gpu_end - t_gpu_start;

	//simple_sgemm_eigen(h_A, h_B, h_C_ref);

	//printf("N=%4d, GPU=%.6fms(%.3fGflops)\n",
	//	N, gpu_t, 1e-9*N*N*N * 2 / gpu_t);

	///* 检查CPU结果与GPU结果是否相等 */
	//float ref_norm = 0.;
	//float error_norm = 0.;
	//for (int i = 0; i < N*N; i++) {
	//	float diff = h_C_ref[i] - h_C[i];
	//	error_norm += diff*diff;
	//	ref_norm += h_C_ref[i] * h_C_ref[i];
	//}
	//printf("Test %s\n", (sqrtf(error_norm / ref_norm) < 1e-6) ? "PASSED" : "FAILED");

	system("pause");
	return 0;
}