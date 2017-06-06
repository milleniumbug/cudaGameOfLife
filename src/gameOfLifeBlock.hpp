#pragma once
#include "cudaUtils.hpp"
#include "common.hpp"
#include <utility>
#include <array>

class GameOfLifeBlock
{
	mutable SynchronizedPrimitiveBuffer<bool> central;
	SynchronizedPrimitiveBuffer<bool> next;
	SynchronizedPrimitiveBuffer<bool> borderCheck;
	SynchronizedPrimitiveBuffer<const bool*> cudaSurrounding;
	mutable unsigned synchronized;
	bool commited;
	CudaStream stream;

public:
	GameOfLifeBlock();
	std::array<bool, maxNeighbourAndSelfCount> collectBoundaryInfo();
	void nextGeneration(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours);
	void nextGenerationCommit();
	void setAt(std::size_t i, std::size_t j, bool what);
	bool getAt(std::size_t i, std::size_t j) const;
};
