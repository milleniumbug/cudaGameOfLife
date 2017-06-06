#pragma once
#include <iostream>
#include <memory>
#include <cassert>
#include <type_traits>
#include "make.hpp"

template<typename cudaError_t>
void reportCudaError(cudaError_t errorCode);

struct CudaDeleter
{
	void operator()(void* ptr) const;
};

namespace detail
{
	void* cudaCalloc(std::size_t size, std::size_t count);
	void copyToDevice(void* dest, const void* src, std::size_t size);
	void copyToHost(void* dest, const void* src, std::size_t size);
	void cudaZeroOut(void* what, std::size_t size);
}

template<typename T>
std::unique_ptr<T, CudaDeleter> cudaMakeUniqueArray(std::size_t count)
{
	typedef typename std::unique_ptr<T, CudaDeleter>::element_type element_type;
	// lol, broken on VS2015
	//static_assert(std::is_trivially_destructible<element_type>::value, "must be trivially destructible");
	auto untyped = detail::cudaCalloc(sizeof(element_type), count);
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
		hostMemory(wiertlo::make_unique<T[]>(size)),
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

	void copyToDevice()
	{
		detail::copyToDevice(deviceMemory.get(), hostMemory.get(), size_ * sizeof(T));
	}

	void copyToHost()
	{
		detail::copyToHost(hostMemory.get(), deviceMemory.get(), size_ * sizeof(T));
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

template<typename T>
void cudaBzero(SynchronizedPrimitiveBuffer<T>& input)
{
	memset(input.getHost(), 0, sizeof(T) * input.size());
	detail::cudaZeroOut(input.getDevice(), sizeof(T) * input.size());
}

void printComputeCapability();