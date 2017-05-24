
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <array>
#include <iostream>
#include "cudaUtils.cuh"

const int maxNeighbourCount = 8;
const int maxNeighbourAndSelfCount = maxNeighbourCount + 1;
const int center = 4;
const int upOrDown = 2;
const int leftOrRight = 1;
const int blockDimension = 64;

__device__ int wrap(int in)
{
	return (in + blockDimension * 2) % blockDimension;
}

// in < 0: -1
// in >= 0 && in < wrapCounter: 0
// in >= wrapCounter: 1
__device__ int robert(int in)
{
	if(in < 0)
		return -1;
	if(in >= blockDimension)
		return 1;
	return 0;
}

// 0, 0 : 0
// 0, 1 : 0
// 0, 2 : 0
// 0, 3 : 1
// 0, 4 : 0
// 0, 5 : 0
// 0, 6 : 0
// 0, 7 : 0
// 0, 8 : 0
// 0, 9 : X
// 0, 10: X
// 0, 11: X
// 0, 12: X
// 0, 13: X
// 0, 14: X
// 0, 15: X
// 1, 0 : 0
// 1, 1 : 0
// 1, 2 : 1
// 1, 3 : 1
// 1, 4 : 0
// 1, 5 : 0
// 1, 6 : 0
// 1, 7 : 0
// 1, 8 : 0
// 1, 9 : X
// 1, 10: X
// 1, 11: X
// 1, 12: X
// 1, 13: X
// 1, 14: X
// 1, 15: X
__device__ bool rule(bool current, int neighbourCount)
{
	if(neighbourCount == 2)
		return current;
	else if(neighbourCount == 3)
		return true;
	else
		return false;
}

__global__ void nextGeneration(bool* next_generation, const bool* const* surrounding, bool* out)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= blockDimension || y >= blockDimension)
		return;

	int neighbourCount =
		surrounding[center + robert(x - 1) * leftOrRight + robert(y - 1) * upOrDown][wrap(x - 1) + wrap(y - 1)*blockDimension] +
		surrounding[center +                               robert(y - 1) * upOrDown][x +           wrap(y - 1)*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight + robert(y - 1) * upOrDown][wrap(x + 1) + wrap(y - 1)*blockDimension] +

		surrounding[center + robert(x - 1) * leftOrRight                           ][wrap(x - 1) + y*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight                           ][wrap(x + 1) + y*blockDimension] +

		surrounding[center + robert(x - 1) * leftOrRight + robert(y + 1) * upOrDown][wrap(x - 1) + wrap(y + 1)*blockDimension] +
		surrounding[center +                               robert(y + 1) * upOrDown][x +           wrap(y + 1)*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight + robert(y + 1) * upOrDown][wrap(x + 1) + wrap(y + 1)*blockDimension];

	next_generation[x + y*blockDimension] = rule(surrounding[center][x + y*blockDimension], neighbourCount);

	if(x == 0 && neighbourCount > 0)
		out[center - leftOrRight] = true;
	if(x == blockDimension - 1 && neighbourCount > 0)
		out[center + leftOrRight] = true;
	if(y == 0 && neighbourCount > 0)
		out[center - upOrDown] = true;
	if(y == blockDimension - 1 && neighbourCount > 0)
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
	nextGeneration <<< dimensions, threadsPerBlock >>>(central.getDevice(), cudaSurrounding.getDevice(), borderCheck.getDevice());
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