#pragma once

#include <iostream>

#include "context.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"

class ContextImpl final : public Context {
public:
	ContextImpl();
	~ContextImpl() override;
	ContextImpl(const ContextImpl&) = delete;
	void operator=(const ContextImpl&) = delete;

	void EnsureMinimumThreads(int num_threads);

	// Initialize the cuSolverDn context, creates an asynchronous stream, and
	// associate the stream with cuSolverDn, Return true if initialization was
	// successful, else it returns false and a human-readable error message is 
	// returned.
	bool InitCUDA(std::string* message);

	// Handle to the cuSolver context.
	cusolverDnHandle_t cusolver_handle_ = nullptr;
	// Handle to cuBlas context.
	cublasHandle_t cublas_handle_ = nullptr;
	// CUDA device stream.
	cudaStream_t stream_ = nullptr;
	// Indicates whether all the CUDA resource have been initialized.
	bool cuda_initialized_ = false;
};
