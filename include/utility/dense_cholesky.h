#pragma once

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "cuda_runtime.h"
#include "cusolverDn.h"

#include "utility/cuda_buffer.h"
#include "utility/context_impl.h"

struct Options {
	Options() = default;
	ContextImpl* context;
};

enum LinearSolverTerminationType {
	FATAL_ERROR = -1,
	SUCCESS,
	FAILURE
};

class DenseCholesky {
public:
	static std::unique_ptr<DenseCholesky> Create(
		const Options& options);

	virtual ~DenseCholesky();

	virtual int Factorize(int num_cols, 
						double* lhs, 
						std::string* message) = 0;

	virtual int Solve(const double* rhs, 
					double* solution, 
					std::string* message) = 0;

	int FactorAndSolve(int num_cols, 
					double* lhs, 
					const double* rhs, 
					double* solution, 
					std::string* message);
};

class CUDADenseCholesky final : public DenseCholesky {
public:
	static std::unique_ptr<CUDADenseCholesky> Create(
		const Options& options);
	CUDADenseCholesky(const CUDADenseCholesky&) = delete;
	CUDADenseCholesky& operator=(const CUDADenseCholesky&) = delete;
	int Factorize(int num_cols, double* lhs, std::string* message) override;
	int Solve(const double* rhs, double* solution, std::string* message) override;
private:
	CUDADenseCholesky() = default;
	
	// Picks up the cuSolverDn and cuStream handles from the context. if 
	// the context is unable to initialize CUDA, returns false with a
	// human-readable message indicating the reason.
	bool Init(ContextImpl* context, std::string* message);

	// Handle to the cuSolver context.
	cusolverDnHandle_t cusolver_handle_ = nullptr;
	// CUDA device stream.
	cudaStream_t stream_ = nullptr;
	// Number of colums in the A matrix, to be cached between calls to *Factoize
	// and *Solve.
	size_t num_cols_ = 0;
	// GPU memory allocated for the A matrix (lhs matrix).
	CudaBuffer<double> lhs_;
	// GPU memory allocated for the B matrix (rhs matrix).
	CudaBuffer<double> rhs_;
	// Scratch space for cuSolver on the GPU.
	CudaBuffer<double> device_workspace_;
	// Required for error handling with cuSolver
	CudaBuffer<int> error_;
	// Cache the result of fatorize to ensure that when solve is called, the
	// factorization of lhs is valid
	int factorize_result_ = -1;
};