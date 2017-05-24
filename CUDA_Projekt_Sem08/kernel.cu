
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <array>
#include <iostream>
#include "cudaUtils.cuh"

__device__ int wrap(int in, int wrapCounter)
{
	return (in + wrapCounter * 2) % wrapCounter;
}

// in < 0: -1
// in >= 0 && in < wrapCounter: 0
// in >= wrapCounter: 1
__device__ int robert(int in, int wrapCounter)
{
	if(in < 0)
		return -1;
	if(in >= wrapCounter)
		return 1;
	return 0;
}

__device__ bool rule(bool current, int neighbourCount)
{
	if(neighbourCount == 2)
		return current;
	else if(neighbourCount == 3)
		return true;
	else
		return false;
}

const int maxNeighbourCount = 8;
const int maxNeighbourAndSelfCount = maxNeighbourCount + 1;
const int central = 4;
const int upOrDown = 2;
const int leftOrRight = 1;

__global__ void nextGeneration(bool* next_generation, const bool* const* surrounding, int dim, bool* out)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim || y >= dim)
		return;

	int neighbourCount =
		surrounding[central + robert(x - 1, dim) * leftOrRight + robert(y - 1, dim) * upOrDown][wrap(x - 1, dim) + wrap(y - 1, dim)*dim] +
		surrounding[central +                                    robert(y - 1, dim) * upOrDown][x +                wrap(y - 1, dim)*dim] +
		surrounding[central + robert(x + 1, dim) * leftOrRight + robert(y - 1, dim) * upOrDown][wrap(x + 1, dim) + wrap(y - 1, dim)*dim] +

		surrounding[central + robert(x - 1, dim) * leftOrRight                                ][wrap(x - 1, dim) + y*dim] +
		surrounding[central + robert(x + 1, dim) * leftOrRight                                ][wrap(x + 1, dim) + y*dim] +

		surrounding[central + robert(x - 1, dim) * leftOrRight + robert(y + 1, dim) * upOrDown][wrap(x - 1, dim) + wrap(y + 1, dim)*dim] +
		surrounding[central +                                    robert(y + 1, dim) * upOrDown][x +                wrap(y + 1, dim)*dim] +
		surrounding[central + robert(x + 1, dim) * leftOrRight + robert(y + 1, dim) * upOrDown][wrap(x + 1, dim) + wrap(y + 1, dim)*dim];

	next_generation[x + y*dim] = rule(surrounding[central][x + y*dim], neighbourCount);

	if(x == 0 && neighbourCount > 0)
		out[central - leftOrRight] = true;
	if(x == dim - 1 && neighbourCount > 0)
		out[central + leftOrRight] = true;
	if(y == 0 && neighbourCount > 0)
		out[central - upOrDown] = true;
	if(y == dim - 1 && neighbourCount > 0)
		out[central + upOrDown] = true;
}

int main()
{
	int dim = 64;
	int bufsize = dim*dim;
	dim3 threadsPerBlock(16, 16);
	dim3 dimensions(dim/threadsPerBlock.x, dim/threadsPerBlock.y);

	auto hostCentral = std::make_unique<bool[]>(bufsize);
	auto cudaCentral = cudaMakeUniqueArray<bool[]>(bufsize);
	auto hostBorderCheck = std::make_unique<bool[]>(maxNeighbourAndSelfCount);
	auto cudaBorderCheck = cudaMakeUniqueArray<bool[]>(maxNeighbourAndSelfCount);

	auto hostSource = std::make_unique<bool[]>(bufsize);
	// create a glider
	hostSource[11 + 10 * dim] = true;
	hostSource[12 + 11 * dim] = true;
	hostSource[10 + 12 * dim] = true;
	hostSource[11 + 12 * dim] = true;
	hostSource[12 + 12 * dim] = true;
	std::array<std::unique_ptr<bool[], CudaDeleter>, maxNeighbourAndSelfCount> cudaSurroundingPtrs =
	{
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
		cudaMakeUniqueArray<bool[]>(bufsize),
	};
	reportCudaError(cudaMemcpy(cudaSurroundingPtrs[central].get(), hostSource.get(), bufsize * sizeof cudaSurroundingPtrs[central][0], cudaMemcpyHostToDevice));
	
	auto hostCudaSurrounding = std::make_unique<bool*[]>(maxNeighbourAndSelfCount);
	auto cudaCudaSurrounding = cudaMakeUniqueArray<bool*[]>(maxNeighbourAndSelfCount);
	for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		hostCudaSurrounding[i] = cudaSurroundingPtrs[i].get();
	}
	reportCudaError(cudaMemcpy(cudaCudaSurrounding.get(), hostCudaSurrounding.get(), maxNeighbourAndSelfCount * sizeof cudaCudaSurrounding[0], cudaMemcpyHostToDevice));

	reportCudaError(cudaMemcpy(cudaCentral.get(), hostCentral.get(), bufsize * sizeof cudaCentral[0], cudaMemcpyHostToDevice));
	nextGeneration <<< dimensions, threadsPerBlock >>>(cudaCentral.get(), cudaCudaSurrounding.get(), dim, cudaBorderCheck.get());
	reportCudaError(cudaMemcpy(hostCentral.get(), cudaCentral.get(), bufsize * sizeof cudaCentral[0], cudaMemcpyDeviceToHost));

	reportCudaError(cudaMemcpy(hostBorderCheck.get(), cudaBorderCheck.get(), maxNeighbourAndSelfCount * sizeof hostBorderCheck[0], cudaMemcpyDeviceToHost));

	for(int j = 0; j < dim; ++j)
	{
		for(int i = 0; i < dim; ++i)
		{
			std::cout << (hostCentral[j*dim + i] ? "X" : " ");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	for(int i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		if(hostBorderCheck[i])
			std::cout << i;
		else
			std::cout << " ";
	}
	std::cout << "\n";
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
