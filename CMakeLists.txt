cmake_minimum_required(VERSION 2.8)
project(game_of_life)

find_package(CUDA)
cuda_add_executable(game_of_life
	src/bitset_view.cuh
	src/common.cpp
	src/common.hpp
	src/config.hpp
	src/cudaUtils.hpp
	src/cudaUtils.cu
	src/gameOfLife.cpp
	src/gameOfLife.hpp
	src/gameOfLifeBlock.cu
	src/gameOfLifeBlock.hpp
	src/main.cpp
	src/make.hpp
	src/sampleBoards.cpp
	src/sampleBoards.hpp
	src/test.cpp)

# fucking CMake 2.8
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")