#pragma once
#include <utility>

typedef std::pair<int, int> position_type;

const int maxNeighbourCount = 8;
const int maxNeighbourAndSelfCount = maxNeighbourCount + 1;
const int center = 4;
const int upOrDown = 3;
const int leftOrRight = 1;

position_type shift(position_type position, int direction);