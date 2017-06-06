#pragma once
#include <cstddef>
#include <host_defines.h>

template<typename Integral>
class bitset_view
{
	Integral* underlying;

public:

	class reference
	{
		Integral* underlying;
		Integral bit;

		friend class bitset_view<Integral>;

		__host__ __device__ reference(Integral* underlying, Integral bit) :
			underlying(underlying),
			bit(bit)
		{

		}

	public:
		__host__ __device__ reference& operator=(bool x)
		{
			*underlying |= x ? bit : static_cast<Integral>(0);
			return *this;
		}

		__host__ __device__ operator bool() const
		{
			return (*underlying & bit) != 0;
		}

		__host__ __device__ bool operator~() const
		{
			return ~(operator bool());
		}
	};

	__host__ __device__ reference operator[](std::size_t pos)
	{
		return reference(underlying, static_cast<Integral>(1) << pos);
	}

	__host__ __device__ bool operator[](std::size_t pos) const
	{
		return reference(underlying, static_cast<Integral>(1) << pos);
	}

	__host__ __device__ bitset_view(Integral* underlying) :
		underlying(underlying)
	{

	}
};

template<typename Integral>
__host__ __device__ bitset_view<Integral> bitview(Integral& what)
{
	return bitset_view<Integral>(&what);
}