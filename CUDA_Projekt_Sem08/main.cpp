
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
#include <chrono>

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

template<typename Duration>
std::string adaptiveStringFromTime(Duration duration)
{
	std::string base = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) + " microseconds (";
	if(duration > std::chrono::milliseconds(100000))
	{
		return base + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(duration).count()) + " seconds)";
	}
	else if(duration > std::chrono::microseconds(100000))
	{
		return base + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()) + " milliseconds)";
	}
	else if(duration > std::chrono::nanoseconds(100000))
	{
		return base + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) + " microseconds)";
	}
	else
	{
		return base + std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()) + " nanoseconds)";
	}
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
				std::string modeStr;
				RunMode mode = RunMode::Gpu;
				if(ss >> number)
				{
					if(ss >> modeStr)
					{
						if(modeStr == "CPU")
							mode = RunMode::Cpu;
						if(modeStr == "GPU")
							mode = RunMode::Gpu;
					}
					auto before = std::chrono::high_resolution_clock::now();
					for(int i = 0; i < number; ++i)
					{
						game.nextGeneration(mode);
					}
					auto after = std::chrono::high_resolution_clock::now();
					std::cout << "Executed in: " << adaptiveStringFromTime(after - before) << "\n";
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