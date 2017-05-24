
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
#include "gameOfLifeBlock.hpp"
#include "config.hpp"
#include "gameOfLife.hpp"

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