#include "gameOfLifeBlock.hpp"
#include "config.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const dim3 threadsPerBlock(threadsPerDimension, threadsPerDimension);
const dim3 dimensions(blockDimension / threadsPerBlock.x, blockDimension / threadsPerBlock.y);

__host__ __device__ int wrap(int in)
{
	// thanks https://stackoverflow.com/a/44184152/1012936
	return ((in % blockDimension) + blockDimension) % blockDimension;
}

// in < 0: -1
// in >= 0 && in < wrapCounter: 0
// in >= wrapCounter: 1
__host__ __device__ int robert(int in)
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
__host__ __device__ bool rule(bool current, int neighbourCount)
{
	if(neighbourCount == 2)
		return current;
	else if(neighbourCount == 3)
		return true;
	else
		return false;
}

__host__ __device__ void nextGenerationKernelImpl(bool* next_generation, const bool* const* surrounding, bool* out, int x, int y)
{
	if(x >= blockDimension || y >= blockDimension)
		return;

	int neighbourCount =
		surrounding[center + robert(x - 1) * leftOrRight + robert(y - 1) * upOrDown][wrap(x - 1) + wrap(y - 1)*blockDimension] +
		surrounding[center + robert(y - 1) * upOrDown][x + wrap(y - 1)*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight + robert(y - 1) * upOrDown][wrap(x + 1) + wrap(y - 1)*blockDimension] +

		surrounding[center + robert(x - 1) * leftOrRight][wrap(x - 1) + y*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight][wrap(x + 1) + y*blockDimension] +

		surrounding[center + robert(x - 1) * leftOrRight + robert(y + 1) * upOrDown][wrap(x - 1) + wrap(y + 1)*blockDimension] +
		surrounding[center + robert(y + 1) * upOrDown][x + wrap(y + 1)*blockDimension] +
		surrounding[center + robert(x + 1) * leftOrRight + robert(y + 1) * upOrDown][wrap(x + 1) + wrap(y + 1)*blockDimension];

	next_generation[x + y*blockDimension] = rule(surrounding[center][x + y*blockDimension], neighbourCount);

	if(neighbourCount > 0)
	{
		out[center] = true;
		if(x == 0 && y == 0)
			out[center - leftOrRight - upOrDown] = true;
		if(y == 0)
			out[center - upOrDown] = true;
		if(x == blockDimension - 1 && y == 0)
			out[center + leftOrRight - upOrDown] = true;
		if(x == 0)
			out[center - leftOrRight] = true;
		if(x == blockDimension - 1)
			out[center + leftOrRight] = true;
		if(x == 0 && y == blockDimension - 1)
			out[center - leftOrRight + upOrDown] = true;
		if(y == blockDimension - 1)
			out[center + upOrDown] = true;
		if(x == blockDimension - 1 && y == blockDimension - 1)
			out[center + leftOrRight + upOrDown] = true;
	}
}

__global__ void nextGenerationKernel(bool* next_generation, const bool* const* surrounding, bool* out)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	nextGenerationKernelImpl(next_generation, surrounding, out, x, y);
}

GameOfLifeBlock::GameOfLifeBlock() :
	central(blockDimension*blockDimension),
	next(blockDimension*blockDimension),
	borderCheck(maxNeighbourAndSelfCount),
	cudaSurrounding(maxNeighbourAndSelfCount),
	synchronized(cudaMemcpyHostToHost), // cudaMemcpyHostToHost means it's synchronized
	commited(true),
	stream(
#ifdef UNOPTIMIZED
		nullptr
#endif
	)
{

}

std::array<bool, maxNeighbourAndSelfCount> GameOfLifeBlock::collectBoundaryInfo()
{
	borderCheck.copyToHostAsync(stream);
	stream.wait();
	std::array<bool, maxNeighbourAndSelfCount> result;
	for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		result[i] = borderCheck[i];
	}
	return result;
}

void GameOfLifeBlock::nextGeneration(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours)
{
	if(!commited)
		return;

	if(synchronized == cudaMemcpyHostToDevice)
	{
		central.copyToDevice();
		synchronized = cudaMemcpyHostToHost;
	}
	cudaBzeroAsync(borderCheck, stream);

	auto toDev = [&]()
	{
		for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
		{
			cudaSurrounding[i] = neighbours[i]->central.getDevice();
		}
		cudaSurrounding[center] = central.getDevice();
		cudaSurrounding.copyToDeviceAsync(stream);
	};
	toDev();
	nextGenerationKernel <<< dimensions, threadsPerBlock, 0, static_cast<cudaStream_t>(stream.get()) >>> (next.getDevice(), cudaSurrounding.getDevice(), borderCheck.getDevice());
#ifdef UNOPTIMIZED
	collectBoundaryInfo();
#endif
	synchronized = cudaMemcpyDeviceToHost;
	commited = false;
}

void GameOfLifeBlock::setAt(std::size_t i, std::size_t j, bool what)
{
	if(synchronized == cudaMemcpyDeviceToHost)
	{
		central.copyToHost();
		synchronized = cudaMemcpyHostToHost;
	}

	central[j*blockDimension + i] = what;

	synchronized = cudaMemcpyHostToDevice;
}

bool GameOfLifeBlock::getAt(std::size_t i, std::size_t j) const
{
	if(synchronized == cudaMemcpyDeviceToHost)
	{
		central.copyToHost();
		synchronized = cudaMemcpyHostToHost;
	}

	return central[j*blockDimension + i];
}

void GameOfLifeBlock::nextGenerationCommit()
{
	if(!commited)
	{
		stream.wait();
		std::swap(central, next);
	}
	commited = true;
}