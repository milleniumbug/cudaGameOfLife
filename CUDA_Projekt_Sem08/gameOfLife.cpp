#include "gameOfLife.hpp"
#include "gameOfLifeBlock.hpp"
#include "config.hpp"

bool GameOfLife::isEmptyBlock(const GameOfLifeBlock* input)
{
	return input == &emptyBlock;
}

const GameOfLifeBlock* GameOfLife::getAt(position_type pos)
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

std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount> GameOfLife::getNeighbours(const blocks_type::value_type& x)
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

void GameOfLife::materializeAt(position_type pos)
{
	auto block = cachedEmptyBlocks.empty() ? GameOfLifeBlock() : std::move(cachedEmptyBlocks.back());
	if(!cachedEmptyBlocks.empty())
		cachedEmptyBlocks.pop_back();
	materializationRequests.emplace_back(pos, std::move(block));
}

void GameOfLife::dematerializeAt(position_type pos)
{
	dematerializationRequests.push_back(pos);
}

void GameOfLife::materializationCommit()
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

void GameOfLife::simulateRoundFor(blocks_type::value_type& kvp)
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
	if(std::none_of(borders.begin(), borders.end(), [](bool x)
	{
		return x;
	}))
	{
		dematerializeAt(position);
	}
}

void GameOfLife::nextGeneration()
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

std::vector<std::vector<bool>> GameOfLife::dumpStateAt(position_type at)
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

void GameOfLife::setStateAt(position_type at, const std::vector<std::vector<bool>>& what)
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

GameOfLife::GameOfLife()
{

}