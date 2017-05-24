
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
#include <string>
#include <sstream>

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
	auto dump = game.dumpStateAt(pos);
	std::cout << pos.first << " " << pos.second << "\n";
	for(int j = 0; j < blockDimension; ++j)
	{
		for(int i = 0; i < blockDimension; ++i)
		{
			std::cout << (dump[i][j] ? "X" : " ");
		}
		std::cout << "|" << j << "\n";
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

void inputLoop(GameOfLife& game)
{
	std::string line;
	while(std::getline(std::cin, line))
	{
		std::stringstream ss(line);
		std::string command;
		if(ss >> command)
		{
			if(command == "P")
			{
				position_type p;
				if(ss >> p.first >> p.second)
				{
					printAtNear(game, p);
				}
			}
			if(command == "N")
			{
				int number;
				if(ss >> number)
				{
					for(int i = 0; i < number; ++i)
					{
						game.nextGeneration();
					}
				}
			}
		}
	}
}

int main()
{
	GameOfLife game = simpleGliderGame();
	inputLoop(game);
}