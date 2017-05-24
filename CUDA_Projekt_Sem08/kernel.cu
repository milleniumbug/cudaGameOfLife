
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
const dim3 threadsPerBlock(16, 16);
const dim3 dimensions(blockDimension / threadsPerBlock.x, blockDimension / threadsPerBlock.y);

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

__global__ void nextGenerationKernel(bool* next_generation, const bool* const* surrounding, bool* out)
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

class GameOfLifeBlock
{
	SynchronizedPrimitiveBuffer<bool> central;
	SynchronizedPrimitiveBuffer<bool> next;
	SynchronizedPrimitiveBuffer<bool> borderCheck;
	SynchronizedPrimitiveBuffer<const bool*> cudaSurrounding;
	cudaMemcpyKind synchronized;

public:
	GameOfLifeBlock() :
		central(blockDimension*blockDimension),
		next(blockDimension*blockDimension),
		borderCheck(maxNeighbourAndSelfCount),
		cudaSurrounding(maxNeighbourAndSelfCount),
		synchronized(cudaMemcpyHostToHost) // cudaMemcpyHostToHost means it's synchronized
	{
		
	}

	std::array<bool, maxNeighbourAndSelfCount> nextGeneration(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours)
	{
		if(synchronized == cudaMemcpyHostToDevice)
		{
			central.copyToDevice();
			synchronized = cudaMemcpyHostToHost;
		}
		cudaBzero(borderCheck);

		auto toDev = [&]()
		{
			for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
			{
				cudaSurrounding[i] = neighbours[i]->central.getDevice();
			}
			cudaSurrounding[center] = central.getDevice();
			cudaSurrounding.copyToDevice();
		};
		toDev();
		nextGenerationKernel <<< dimensions, threadsPerBlock >>> (next.getDevice(), cudaSurrounding.getDevice(), borderCheck.getDevice());
		std::swap(central, next);
		borderCheck.copyToHost();
		std::array<bool, maxNeighbourAndSelfCount> result;
		for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
		{
			result[i] = borderCheck[i];
		}

		synchronized = cudaMemcpyDeviceToHost;
		return result;
	}

	void setAt(std::size_t i, std::size_t j, bool what)
	{
		if(synchronized == cudaMemcpyDeviceToHost)
		{
			central.copyToHost();
			synchronized = cudaMemcpyHostToHost;
		}

		central[j*blockDimension + i] = what;

		synchronized = cudaMemcpyHostToDevice;
	}

	bool getAt(std::size_t i, std::size_t j)
	{
		if(synchronized == cudaMemcpyDeviceToHost)
		{
			central.copyToHost();
			synchronized = cudaMemcpyHostToHost;
		}

		return central[j*blockDimension + i];
	}
};

int main()
{
	std::array<GameOfLifeBlock, maxNeighbourAndSelfCount> surrounding;
	// create glider
	surrounding[center].setAt(11, 10, true);
	surrounding[center].setAt(12, 11, true);
	surrounding[center].setAt(10, 12, true);
	surrounding[center].setAt(11, 12, true);
	surrounding[center].setAt(12, 12, true);

	std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount> surroundingIn;
	std::transform(surrounding.begin(), surrounding.end(), surroundingIn.begin(), [](auto& x)
	{
		return &x;
	});
	std::array<bool, maxNeighbourAndSelfCount> borders;
	for(int i = 0; i < 1000; ++i)
		borders = surrounding[center].nextGeneration(surroundingIn);

	for(int j = 0; j < blockDimension; ++j)
	{
		for(int i = 0; i < blockDimension; ++i)
		{
			std::cout << (surrounding[center].getAt(i, j) ? "X" : " ");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	for(int i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		if(borders[i])
			std::cout << i;
		else
			std::cout << " ";
	}
	std::cout << "\n";
}