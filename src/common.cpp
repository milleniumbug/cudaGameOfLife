#include "common.hpp"
#include <cassert>

position_type shift(position_type position, int direction)
{
	assert(direction >= 0 && direction <= 8);
	switch(direction)
	{
	case 0:
		return position_type(position.first - 1, position.second - 1);
	case 1:
		return position_type(position.first, position.second - 1);
	case 2:
		return position_type(position.first + 1, position.second - 1);
	case 3:
		return position_type(position.first - 1, position.second);
	case center:
		return position;
	case 5:
		return position_type(position.first + 1, position.second);
	case 6:
		return position_type(position.first - 1, position.second + 1);
	case 7:
		return position_type(position.first, position.second + 1);
	case 8:
		return position_type(position.first + 1, position.second + 1);
	default:
		throw "FUCK";
	}
}
