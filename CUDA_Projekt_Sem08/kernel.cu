
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <array>
#include <iostream>
#include <utility>
#include "cudaUtils.cuh"
#include <map>
#include <vector>

const int maxNeighbourCount = 8;
const int maxNeighbourAndSelfCount = maxNeighbourCount + 1;
const int center = 4;
const int upOrDown = 3;
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

	if(neighbourCount > 0)
	{
		out[center] = true;
		if(x == 0 &&                  y == 0)
			out[center - leftOrRight - upOrDown] = true;
		if(                           y == 0)
			out[center               - upOrDown] = true;
		if(x == blockDimension - 1 && y == 0)
			out[center + leftOrRight - upOrDown] = true;
		if(x == 0)
			out[center - leftOrRight] = true;
		if(x == blockDimension - 1)
			out[center + leftOrRight] = true;
		if(x == 0 &&                  y == blockDimension - 1)
			out[center - leftOrRight + upOrDown] = true;
		if(                           y == blockDimension - 1)
			out[center               + upOrDown] = true;
		if(x == blockDimension - 1 && y == blockDimension - 1)
			out[center + leftOrRight + upOrDown] = true;
	}
}

typedef std::pair<int, int> position_type;

position_type shift(position_type position, int direction)
{
	assert(direction >= 0 && direction <= 8);
	switch(direction)
	{
	case 0:
		return position_type(position.first - 1, position.second - 1);
	case 1:
		return position_type(position.first, position.second - 1);
	case 2:
		return position_type(position.first + 1, position.second - 1);
	case 3:
		return position_type(position.first - 1, position.second);
	case center:
		return position;
	case 5:
		return position_type(position.first + 1, position.second);
	case 6:
		return position_type(position.first - 1, position.second + 1);
	case 7:
		return position_type(position.first, position.second + 1);
	case 8:
		return position_type(position.first + 1, position.second + 1);
	default:
		throw "FUCK";
	}
}

class GameOfLifeBlock
{
	mutable SynchronizedPrimitiveBuffer<bool> central;
	SynchronizedPrimitiveBuffer<bool> next;
	SynchronizedPrimitiveBuffer<bool> borderCheck;
	SynchronizedPrimitiveBuffer<const bool*> cudaSurrounding;
	mutable cudaMemcpyKind synchronized;
	bool commited;

public:
	GameOfLifeBlock() :
		central(blockDimension*blockDimension),
		next(blockDimension*blockDimension),
		borderCheck(maxNeighbourAndSelfCount),
		cudaSurrounding(maxNeighbourAndSelfCount),
		synchronized(cudaMemcpyHostToHost), // cudaMemcpyHostToHost means it's synchronized
		commited(true)
	{
		
	}

	std::array<bool, maxNeighbourAndSelfCount> bordersToHost()
	{
		borderCheck.copyToHost();
		std::array<bool, maxNeighbourAndSelfCount> result;
		for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
		{
			result[i] = borderCheck[i];
		}
		return result;
	}

	std::array<bool, maxNeighbourAndSelfCount> nextGeneration(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours)
	{
		if(!commited)
			return bordersToHost();

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
		auto result = bordersToHost();

		synchronized = cudaMemcpyDeviceToHost;
		commited = false;
		return result;
	}

	void nextGenerationCommit()
	{
		if(!commited)
			std::swap(central, next);
		commited = true;
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

	bool getAt(std::size_t i, std::size_t j) const
	{
		if(synchronized == cudaMemcpyDeviceToHost)
		{
			central.copyToHost();
			synchronized = cudaMemcpyHostToHost;
		}

		return central[j*blockDimension + i];
	}
};

void printBorder(const std::array<bool, maxNeighbourAndSelfCount>& borders)
{
	for(int i = 0; i < maxNeighbourAndSelfCount; ++i)
	{
		if(borders[i])
			std::cout << i;
		else
			std::cout << " ";
	}
}

class GameOfLife
{
	typedef std::map<position_type, GameOfLifeBlock> blocks_type;
	GameOfLifeBlock emptyBlock;
	blocks_type blocks;
	std::vector<GameOfLifeBlock> cachedEmptyBlocks;
	std::vector<blocks_type::value_type> materializationRequests;
	std::vector<blocks_type::key_type> dematerializationRequests;

	bool isEmptyBlock(const GameOfLifeBlock* input)
	{
		return input == &emptyBlock;
	}

	const GameOfLifeBlock* getAt(position_type pos)
	{
		auto it = blocks.find(pos);
		if(it != blocks.end())
		{
			return &it->second;
		}
		else
		{
			return &emptyBlock;
		}
	}

	std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount> getNeighbours(const blocks_type::value_type& x)
	{
		auto& position = x.first;
		std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount> result = 
		{
			getAt(shift(position, 0)),
			getAt(shift(position, 1)),
			getAt(shift(position, 2)),
			getAt(shift(position, 3)),
			&x.second,
			getAt(shift(position, 5)),
			getAt(shift(position, 6)),
			getAt(shift(position, 7)),
			getAt(shift(position, 8)),
		};
		return result;
	}

	void materializeAt(position_type pos)
	{
		auto block = cachedEmptyBlocks.empty() ? GameOfLifeBlock() : std::move(cachedEmptyBlocks.back());
		if(!cachedEmptyBlocks.empty())
			cachedEmptyBlocks.pop_back();
		materializationRequests.emplace_back(pos, std::move(block));
	}

	void dematerializeAt(position_type pos)
	{
		dematerializationRequests.push_back(pos);
	}

	void materializationCommit()
	{
		for(auto& key : dematerializationRequests)
		{
			auto it = blocks.find(key);
			if(it != blocks.end())
			{
				cachedEmptyBlocks.push_back(std::move(it->second));
				blocks.erase(it);
			}
		}
		dematerializationRequests.clear();
		for(auto& kvp : materializationRequests)
		{
			blocks.insert(std::move(kvp));
		}
		materializationRequests.clear();
	}

	void simulateRoundFor(blocks_type::value_type& kvp)
	{
		auto& position = kvp.first;
		auto& block = kvp.second;

		auto neighbours = getNeighbours(kvp);
		auto borders = block.nextGeneration(neighbours);
		for(std::size_t i = 0; i < maxNeighbourAndSelfCount; ++i)
		{
			if(borders[i] && isEmptyBlock(neighbours[i]))
			{
				materializeAt(shift(position, i));
			}
		}
		if(std::none_of(borders.begin(), borders.end(), [](bool x){ return x; }))
		{
			dematerializeAt(position);
		}
	}

	friend class GameOfLifeIterator;

public:
	void nextGeneration()
	{
		for(auto& kvp : blocks)
		{
			simulateRoundFor(kvp);
		}
		while(!materializationRequests.empty() || !dematerializationRequests.empty())
		{
			for(auto& kvp : materializationRequests)
			{
				simulateRoundFor(kvp);
			}
			materializationCommit();
		}
		for(auto& kvp : blocks)
		{
			kvp.second.nextGenerationCommit();
		}
	}

	std::vector<std::vector<bool>> dumpStateAt(position_type at)
	{
		std::vector<std::vector<bool>> dump(blockDimension, std::vector<bool>(blockDimension));
		auto& block = *getAt(at);
		for(int j = 0; j < blockDimension; ++j)
		{
			for(int i = 0; i < blockDimension; ++i)
			{
				dump[i][j] = block.getAt(i, j);
			}
		}
		return dump;
	}

	void setStateAt(position_type at, const std::vector<std::vector<bool>>& what)
	{
		if(isEmptyBlock(getAt(at)))
		{
			materializeAt(at);
			materializationCommit();
		}
		auto& block = blocks.at(at);
		for(int j = 0; j < blockDimension; ++j)
		{
			for(int i = 0; i < blockDimension; ++i)
			{
				block.setAt(i, j, what[i][j]);
			}
		}
	}

	GameOfLife()
	{

	}
};

void printAtNear(GameOfLife& game, position_type pos)
{
	for(int i = 0; i <= 8; ++i)
	{
		auto nextPos = shift(pos, i);
		auto dump = game.dumpStateAt(nextPos);
		std::cout << nextPos.first << " " << nextPos.second << "\n";
		for(int j = 0; j < blockDimension; ++j)
		{
			for(int i = 0; i < blockDimension; ++i)
			{
				std::cout << (dump[i][j] ? "X" : " ");
			}
			std::cout << "|" << j << "\n";
		}
	}
}

GameOfLife simpleGliderGame()
{
	GameOfLife game;
	std::vector<std::vector<bool>> input(blockDimension, std::vector<bool>(blockDimension));
	input[51][10] = true;
	input[52][11] = true;
	input[50][12] = true;
	input[51][12] = true;
	input[52][12] = true;
	game.setStateAt(position_type(0, 0), input);
	return game;
}

int main()
{
	GameOfLife game = simpleGliderGame();
	for(int i = 0; i < 310; ++i)
	{
		game.nextGeneration();
		if(i >= 299)
		{
			printAtNear(game, position_type(0, 0));
		}
	}
}