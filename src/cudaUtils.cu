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

	void copyToDeviceAsync(void* dest, const void* src, std::size_t size, void* stream)
	{
		reportCudaError(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
	}

	void copyToHostAsync(void* dest, const void* src, std::size_t size, void* stream)
	{
		reportCudaError(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
	}

	void cudaZeroOut(void* what, std::size_t size)
	{
		reportCudaError(cudaMemset(what, 0, size));
	}

	void cudaZeroOutAsync(void* what, std::size_t size, void* stream)
	{
		reportCudaError(cudaMemsetAsync(what, 0, size, static_cast<cudaStream_t>(stream)));
	}
}

void printGpuInfo()
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

void CudaStream::CudaStreamDeleter::operator()(void* stream) const
{
	auto raw_stream = static_cast<cudaStream_t>(stream);
	cudaStreamSynchronize(raw_stream);
	cudaStreamDestroy(raw_stream);
}

CudaStream::CudaStream(void* raw)
{
	stream.reset(raw);
}

CudaStream& CudaStream::getDefault()
{
	static CudaStream stream(nullptr);
	return stream;
}

CudaStream::CudaStream()
{
	cudaStream_t raw_stream;
	cudaStreamCreateWithFlags(&raw_stream, cudaStreamDefault);
	stream.reset(raw_stream);
}

void* CudaStream::get()
{
	return stream.get();
}

void CudaStream::wait() const
{
	cudaStreamSynchronize(static_cast<cudaStream_t>(stream.get()));
}
