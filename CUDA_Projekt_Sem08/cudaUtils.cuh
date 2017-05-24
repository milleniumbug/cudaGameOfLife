#pragma once
#include <memory>
#include <cassert>
#include <type_traits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void reportCudaError(cudaError_t errorCode)
{
	assert(errorCode == cudaSuccess);
}

struct CudaDeleter
{
	template<typename T>
	void operator()(T* ptr)
	{
		// lol, broken on VS2015
		//static_assert(std::is_trivially_destructible<T>::value, "must be trivially destructible");
		cudaFree(ptr);
	}
};

template<typename T>
std::unique_ptr<T, CudaDeleter> cudaMakeUniqueArray(std::size_t count)
{
	typedef typename std::unique_ptr<T, CudaDeleter>::element_type element_type;
	// lol, broken on VS2015
	//static_assert(std::is_trivially_destructible<element_type>::value, "must be trivially destructible");
	void* untyped;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&untyped, count * sizeof(element_type));
	reportCudaError(cudaStatus);
	cudaStatus = cudaMemset(untyped, 0, count * sizeof(element_type));
	reportCudaError(cudaStatus);
	auto typed = static_cast<element_type*>(untyped);
	return std::unique_ptr<T, CudaDeleter>(typed);
}

// T: must be trivial
template<typename T>
class SynchronizedPrimitiveBuffer
{
	std::unique_ptr<T[]> hostMemory;
	std::unique_ptr<T[], CudaDeleter> deviceMemory;
	std::size_t size_;
public:
	SynchronizedPrimitiveBuffer(std::size_t size) :
		hostMemory(std::make_unique<T[]>(size)),
		deviceMemory(cudaMakeUniqueArray<T[]>(size)),
		size_(size)
	{
		
	}

	T* getHost()
	{
		return hostMemory.get();
	}

	const T* getHost() const
	{
		return hostMemory.get();
	}

	T* getDevice()
	{
		return deviceMemory.get();
	}

	const T* getDevice() const
	{
		return deviceMemory.get();
	}


	void toDevice()
	{
		reportCudaError(cudaMemcpy(deviceMemory.get(), hostMemory.get(), size_ * sizeof(T), cudaMemcpyHostToDevice));
	}

	void toHost()
	{
		reportCudaError(cudaMemcpy(hostMemory.get(), deviceMemory.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
	}

	std::size_t size() const
	{
		return size_;
	}
	
	T& operator[](std::size_t pos)
	{
		return hostMemory[pos];
	}

	const T& operator[](std::size_t pos) const
	{
		return hostMemory[pos];
	}
};