#pragma once
#include "cudaUtils.cuh"
#include "common.hpp"
#include <utility>
#include <array>

class GameOfLifeBlock
{
	mutable SynchronizedPrimitiveBuffer<bool> central;
	SynchronizedPrimitiveBuffer<bool> next;
	SynchronizedPrimitiveBuffer<bool> borderCheck;
	SynchronizedPrimitiveBuffer<const bool*> cudaSurrounding;
	mutable cudaMemcpyKind synchronized;
	bool commited;

public:
	GameOfLifeBlock();
	std::array<bool, maxNeighbourAndSelfCount> collectBoundaryInfo();
	void nextGenerationCpu(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours);
	void nextGeneration(const std::array<const GameOfLifeBlock*, maxNeighbourAndSelfCount>& neighbours);
	void nextGenerationCommit();
	void setAt(std::size_t i, std::size_t j, bool what);
	bool getAt(std::size_t i, std::size_t j) const;
};
