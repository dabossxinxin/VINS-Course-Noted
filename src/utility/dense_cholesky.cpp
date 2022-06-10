#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "utility/dense_cholesky.h"
#include "utility/context_impl.h"

#include "cuda_runtime.h"
#include "cusolverDn.h"

DenseCholesky::~DenseCholesky() = default;

std::unique_ptr<DenseCholesky> DenseCholesky::Create(
	const Options& options) {
	std::unique_ptr<DenseCholesky> dence_cholesky;
	dence_cholesky = CUDADenseCholesky::Create(options);
	return dence_cholesky;
}

int DenseCholesky::FactorAndSolve(
	int num_cols,
	double* lhs,
	const double* rhs,
	double* solution,
	std::string* message) {
	int termination_type = Factorize(num_cols, lhs, message);
	if (termination_type == LinearSolverTerminationType::SUCCESS) {
		termination_type = Solve(rhs, solution, message);
	}
	return termination_type;
}

bool CUDADenseCholesky::Init(ContextImpl* context, std::string* message) {
	if (!context->InitCUDA(message)) {
		return false;
	}
	cusolver_handle_ = context->cusolver_handle_;
	stream_ = context->stream_;
	error_.Reserve(1);
	*message = "CUDADenseCholesky::Init Success.";
	return true;
}

int CUDADenseCholesky::Factorize(
	int num_cols,
	double* lhs,
	std::string* message) {
	factorize_result_ = LinearSolverTerminationType::FATAL_ERROR;
	lhs_.Reserve(num_cols*num_cols);
	num_cols_ = num_cols;
	lhs_.CopyToGpuAsync(lhs, num_cols*num_cols, stream_);
	int device_workspace_size = 0;
	if (cusolverDnDpotrf_bufferSize(
		cusolver_handle_,
		CUBLAS_FILL_MODE_LOWER,
		num_cols,
		lhs_.data(),
		num_cols,
		&device_workspace_size) !=
		CUBLAS_STATUS_SUCCESS) {
		*message = "cuSolverDN::cusolverDnDpotrf_bufferSize failed.";
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	device_workspace_.Reserve(device_workspace_size);
	if (cusolverDnDpotrf(
		cusolver_handle_,
		CUBLAS_FILL_MODE_LOWER,
		num_cols,
		lhs_.data(),
		num_cols,
		reinterpret_cast<double*>(device_workspace_.data()),
		device_workspace_.size(),
		error_.data()) != CUSOLVER_STATUS_SUCCESS) {
		*message = "cuSolverDN::cusolverDnDpotrf failed.";
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	if (cudaDeviceSynchronize() != cudaSuccess ||
		cudaStreamSynchronize(stream_) != cudaSuccess) {
		*message = "Cuda device synchronization failed.";
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	int error = 0;
	error_.CopyToHost(&error, 1);
	if (error < 0) {
		std::cerr << "Congratulations, you found a bug in Ceres - "
			<< "please report it. "
			<< "cuSolverDN::cusolverDnXpotrf fatal error. "
			<< "Argument: " << -error << " is invalid." << std::endl;
		// The following line is unreachable, but return failure just to be
		// pedantic, since the compiler does not know that.
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	else if (error > 0) {
		*message =
			"cuSolverDN::cusolverDnDpotrf numerical failure. "
			"The leading minor of order %d is not positive definite.";
		factorize_result_ = LinearSolverTerminationType::FAILURE;
		return LinearSolverTerminationType::FAILURE;
	}
	*message = "Success";
	factorize_result_ = LinearSolverTerminationType::SUCCESS;
	return LinearSolverTerminationType::SUCCESS;
}

int CUDADenseCholesky::Solve(
	const double* rhs,
	double* solution,
	std::string* message) {
	if (factorize_result_ != 0) {
		*message = "Factorize did not complete successfully previously.";
		return factorize_result_;
	}
	rhs_.CopyToGpuAsync(rhs, num_cols_, stream_);
	if (cusolverDnDpotrs(
		cusolver_handle_,
		CUBLAS_FILL_MODE_LOWER,
		num_cols_,
		1,
		lhs_.data(),
		num_cols_,
		rhs_.data(),
		num_cols_,
		error_.data()) != CUSOLVER_STATUS_SUCCESS) {
		*message = "cuSolverDN::cusolverDnDpotrs failed.";
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	if (cudaDeviceSynchronize() != cudaSuccess ||
		cudaStreamSynchronize(stream_) != cudaSuccess) {
		*message = "Cuda device synchronization failed.";
		return LinearSolverTerminationType::FATAL_ERROR;
	}
	int error = 0;
	error_.CopyToHost(&error, 1);
	if (error != 0) {
		std::cerr << "Congratulations, you found a bug in Ceres. "
			<< "Please report it."
			<< "cuSolverDN::cusolverDnDpotrs fatal error. "
			<< "Argument: " << -error << " is invalid." << std::endl;
	}
	rhs_.CopyToHost(solution, num_cols_);
	*message = "Success";
	return LinearSolverTerminationType::SUCCESS;
}

std::unique_ptr<CUDADenseCholesky> CUDADenseCholesky::Create(
	const Options& options) {
	auto cuda_dense_cholesky =
		std::unique_ptr<CUDADenseCholesky>(new CUDADenseCholesky());
	std::string cuda_error;
	if (cuda_dense_cholesky->Init(options.context, &cuda_error)) {
		return cuda_dense_cholesky;
	}
	std::cerr << "CUDADenseCholesky::Init failed: " << cuda_error << std::endl;
	return nullptr;
}