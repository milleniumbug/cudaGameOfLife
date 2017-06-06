#include "cudaUtils.hpp"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<>
void reportCudaError<cudaError_t>(cudaError_t errorCode)
{
	if(errorCode != cudaSuccess)
		std::cerr << cudaGetErrorString(errorCode) << "\n";
}

void CudaDeleter::operator()(void* ptr) const
{
	// lol, broken on VS2015
	//static_assert(std::is_trivially_destructible<T>::value, "must be trivially destructible");
	cudaFree(ptr);
}

namespace detail
{
	void* cudaCalloc(std::size_t size, std::size_t count)
	{
		void* untyped;
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc(&untyped, count * size);
		reportCudaError(cudaStatus);
		cudaStatus = cudaMemset(untyped, 0, count * size);
		reportCudaError(cudaStatus);
		return untyped;
	}

	void copyToDevice(void* dest, const void* src, std::size_t size)
	{
		reportCudaError(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
	}

	void copyToHost(void* dest, const void* src, std::size_t size)
	{
		reportCudaError(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
	}

	void cudaZeroOut(void* what, std::size_t size)
	{
		reportCudaError(cudaMemset(what, 0, size));
	}
}

void printComputeCapability()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for(int i = 0; i < nDevices; ++i)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device " << i << " compute capability: " << prop.major << "." << prop.minor << "\n";
	}
}