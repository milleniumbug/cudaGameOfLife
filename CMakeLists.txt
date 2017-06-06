cmake_minimum_required(VERSION 3.3)
project(game_of_life)

find_package(CUDA)
cuda_add_executable(game_of_life
	src/bitset_view.cuh
	src/common.cpp
	src/common.hpp
	src/config.hpp
	src/cudaUtils.cuh
	src/gameOfLife.cpp
	src/gameOfLife.hpp
	src/gameOfLifeBlock.cu
	src/gameOfLifeBlock.hpp
	src/main.cpp
	src/sampleBoards.cpp
	src/sampleBoards.hpp
	src/test.cpp)