#pragma once
#include <vector>
#include <map>
#include "common.hpp"
#include "gameOfLifeBlock.hpp"

class GameOfLife
{
	typedef std::map<position_type, GameOfLifeBlock> blocks_type;
	GameOfLifeBlock emptyBlock;
	blocks_type blocks;
	std::vector<GameOfLifeBlock> cachedEmptyBlocks;
	std::vector<blocks_type::value_type> materializationRequests;
	std::vector<blocks_type::key_type> dematerializationRequests;

	bool isEmptyBlock(const GameOfLifeBlock* input);
	const GameOfLifeBlock* getAt(position_type pos);
	std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount> getNeighbours(const blocks_type::value_type& x);
	void materializeAt(position_type pos);
	void dematerializeAt(position_type pos);
	void materializationCommit();
	void simulateRoundFor(blocks_type::value_type& kvp);
public:
	void nextGeneration();
	std::vector<std::vector<bool>> dumpStateAt(position_type at);
	void setStateAt(position_type at, const std::vector<std::vector<bool>>& what);
	GameOfLife();
};
