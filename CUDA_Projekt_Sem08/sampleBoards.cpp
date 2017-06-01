#include "sampleBoards.hpp"
#include "common.hpp"
#include "config.hpp"
#include <vector>
#include <random>
#include "gameOfLife.hpp"

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

GameOfLife randomBoardOfSize(position_type dimensions)
{
	GameOfLife game;
	std::mt19937 mt;
	//std::bernoulli_distribution is too slow
	std::uniform_int_distribution<> bernoulli(0, 1);
	std::vector<std::vector<bool>> input(blockDimension, std::vector<bool>(blockDimension));
	auto randomBoard = [&]()
	{
		for(int i = 0; i < blockDimension; ++i)
		{
			for(int j = 0; j < blockDimension; ++j)
			{
				input[i][j] = static_cast<bool>(bernoulli(mt));
			}
		}
	};
	for(int j = 0; j < dimensions.second; ++j)
	{
		for(int i = 0; i < dimensions.first; ++i)
		{
			randomBoard();
			game.setStateAt(position_type(i, j), input);
		}
	}
	return game;
}