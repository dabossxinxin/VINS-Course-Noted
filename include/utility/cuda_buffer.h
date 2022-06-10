#pragma once

#include <iostream>

#include "cuda_runtime.h"
#include "glog/logging.h"

template <typename T>
class CudaBuffer{
public:
	CudaBuffer() = default;
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator=(const CudaBuffer&) = delete;

	~CudaBuffer(){
		if (data_!=nullptr){
			CHECK_EQ(cudaFree(data_), cudaSuccess);
		}
	}

	void Reserve(const size_t size){
		if (size > size_) {
			if (data_ != nullptr) {
				CHECK_EQ(cudaFree(data_), cudaSuccess);
			}
			CHECK_EQ(cudaMalloc(&data_, size * sizeof(T)), cudaSuccess);
		}
	}

	void CopyToGpuAsync(const T* data, const size_t size, cudaStream_t stream) {
		Reserve(size);
		CHECK_EQ(cudaMemcpyAsync(
			data_, data, size * sizeof(T), cudaMemcpyHostToDevice, stream), 
			cudaSuccess);
	}

	void CopyToHost(T* data, const size_t size) {
		CHECK(data_ != nullptr);
		CHECK_EQ(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost),
			cudaSuccess);
	}

	void CopyToGpu(const std::vector<T>& data) {
		CopyToGpu(data.data(), data.size());
	}

	T* data() {
		return data_;
	}

	size_t size() const {
		return size_;
	}
private:
	T* data_ = nullptr;
	size_t size_ = 0;
};