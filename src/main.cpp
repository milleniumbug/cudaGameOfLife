
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
#include <random>
#include "sampleBoards.hpp"

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

void printVec(const std::vector<std::vector<bool>>& vec)
{	
	for(int i = 0; i < blockDimension; ++i)
	{
		for(int j = 0; j < blockDimension; ++j)
		{
			std::cout << (vec[i][j] ? "X" : " ");
		}
		std::cout << "|" << i << "\n";
	}
}

void printAtNear(GameOfLife& game, position_type pos)
{
	auto dump = game.dumpStateAt(pos);
	std::cout << pos.first << " " << pos.second << "\n";
	printVec(dump);
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

bool test(const std::string& name, bool notifyOnSuccess);

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
			if(command == "TEST")
			{
				std::string testName;
				if(ss >> testName)
				{
					test(testName, true);
				}
			}
		}
	}
}

void printComputeCapability()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for(int i = 0; i < nDevices; ++i)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device " << i << " compute capability: " << prop.major << "." << prop.minor << "\n";
	}
}

int main()
{
	printComputeCapability();
	std::cout << "Initializing board...\n";
	
	//GameOfLife game = simpleGliderGame();
	GameOfLife game = randomBoardOfSize(position_type(20, 20));
	std::cout << "Initialized.\n";
	inputLoop(game);
}

void reportCudaError(cudaError_t errorCode)
{
	if(errorCode != cudaSuccess)
		std::cerr << cudaGetErrorString(errorCode) << "\n";
}