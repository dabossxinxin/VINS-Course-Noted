#include "utility/context_impl.h"

ContextImpl::ContextImpl() = default;

bool ContextImpl::InitCUDA(std::string* message) {
	if (cuda_initialized_) {
		return true;
	}
	if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
		*message = "cuBLAS::cublasCreate failed.";
		cublas_handle_ = nullptr;
		return false;
	}
	if (cusolverDnCreate(&cusolver_handle_) != CUSOLVER_STATUS_SUCCESS) {
		*message = "cuSolverDN::cusolverDnCreate failed.";
		cusolver_handle_ = nullptr;
		cublasDestroy(cublas_handle_);
		cublas_handle_ = nullptr;
		return false;
	}
	if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
		cudaSuccess) {
		*message = "CUDA::cudaStreamCreateWithFlags failed.";
		cusolverDnDestroy(cusolver_handle_);
		cublasDestroy(cublas_handle_);
		cusolver_handle_ = nullptr;
		cublas_handle_ = nullptr;
		stream_ = nullptr;
		return false;
	}
	if (cusolverDnSetStream(cusolver_handle_, stream_) !=
		CUSOLVER_STATUS_SUCCESS ||
		cublasSetStream(cublas_handle_, stream_) != CUBLAS_STATUS_SUCCESS) {
		*message =
			"cuSolverDN::cusolverDnSetStream or cuBLAS::cublasSetStream failed.";
		cusolverDnDestroy(cusolver_handle_);
		cublasDestroy(cublas_handle_);
		cudaStreamDestroy(stream_);
		cusolver_handle_ = nullptr;
		cublas_handle_ = nullptr;
		stream_ = nullptr;
		return false;
	}
	cuda_initialized_ = true;
	return true;
}

ContextImpl::~ContextImpl() {
	if (cuda_initialized_) {
		cusolverDnDestroy(cusolver_handle_);
		cublasDestroy(cublas_handle_);
		cudaStreamDestroy(stream_);
	}
}

void ContextImpl::EnsureMinimumThreads(int num_threads) {
	
}