
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
const int center = 4;
const int upOrDown = 2;
const int leftOrRight = 1;

__global__ void nextGeneration(bool* next_generation, const bool* const* surrounding, int dim, bool* out)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim || y >= dim)
		return;

	int neighbourCount =
		surrounding[center + robert(x - 1, dim) * leftOrRight + robert(y - 1, dim) * upOrDown][wrap(x - 1, dim) + wrap(y - 1, dim)*dim] +
		surrounding[center +                                    robert(y - 1, dim) * upOrDown][x +                wrap(y - 1, dim)*dim] +
		surrounding[center + robert(x + 1, dim) * leftOrRight + robert(y - 1, dim) * upOrDown][wrap(x + 1, dim) + wrap(y - 1, dim)*dim] +

		surrounding[center + robert(x - 1, dim) * leftOrRight                                ][wrap(x - 1, dim) + y*dim] +
		surrounding[center + robert(x + 1, dim) * leftOrRight                                ][wrap(x + 1, dim) + y*dim] +

		surrounding[center + robert(x - 1, dim) * leftOrRight + robert(y + 1, dim) * upOrDown][wrap(x - 1, dim) + wrap(y + 1, dim)*dim] +
		surrounding[center +                                    robert(y + 1, dim) * upOrDown][x +                wrap(y + 1, dim)*dim] +
		surrounding[center + robert(x + 1, dim) * leftOrRight + robert(y + 1, dim) * upOrDown][wrap(x + 1, dim) + wrap(y + 1, dim)*dim];

	next_generation[x + y*dim] = rule(surrounding[center][x + y*dim], neighbourCount);

	if(x == 0 && neighbourCount > 0)
		out[center - leftOrRight] = true;
	if(x == dim - 1 && neighbourCount > 0)
		out[center + leftOrRight] = true;
	if(y == 0 && neighbourCount > 0)
		out[center - upOrDown] = true;
	if(y == dim - 1 && neighbourCount > 0)
		out[center + upOrDown] = true;
}

int main()
{
	int dim = 64;
	int bufsize = dim*dim;
	dim3 threadsPerBlock(16, 16);
	dim3 dimensions(dim/threadsPerBlock.x, dim/threadsPerBlock.y);

	SynchronizedPrimitiveBuffer<bool> central(bufsize);
	SynchronizedPrimitiveBuffer<bool> borderCheck(maxNeighbourAndSelfCount);

	auto hostSource = std::make_unique<bool[]>(bufsize);
	std::array<SynchronizedPrimitiveBuffer<bool>, maxNeighbourAndSelfCount> cudaSurroundingPtrs =
	{
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
		SynchronizedPrimitiveBuffer<bool>(bufsize),
	};
	// create a glider
	cudaSurroundingPtrs[center][11 + 10 * dim] = true;
	cudaSurroundingPtrs[center][12 + 11 * dim] = true;
	cudaSurroundingPtrs[center][10 + 12 * dim] = true;
	cudaSurroundingPtrs[center][11 + 12 * dim] = true;
	cudaSurroundingPtrs[center][12 + 12 * dim] = true;
	cudaSurroundingPtrs[center].toDevice();
	
	SynchronizedPrimitiveBuffer<bool*> cudaSurrounding(maxNeighbourAndSelfCount);
	for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		cudaSurrounding[i] = cudaSurroundingPtrs[i].getDevice();
	}
	cudaSurrounding.toDevice();

	central.toDevice();
	nextGeneration <<< dimensions, threadsPerBlock >>>(central.getDevice(), cudaSurrounding.getDevice(), dim, borderCheck.getDevice());
	central.toHost();

	borderCheck.toHost();

	for(int j = 0; j < dim; ++j)
	{
		for(int i = 0; i < dim; ++i)
		{
			std::cout << (central[j*dim + i] ? "X" : " ");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	for(int i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		if(borderCheck[i])
			std::cout << i;
		else
			std::cout << " ";
	}
	std::cout << "\n";
}