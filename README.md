cudaGameOfLife
==============

Exactly what it says on the tin.


Current inner workings
----------------------

My board is split up in square "blocks", and each one of them manages memory allocated by CUDA - "next" and "current" block. Currently, each cell is a `bool` - bit operations would be way better, I think. Each block has a size of 64x64=4096 cells. I provide the following to the kernel:
 - pointer to contiguous "flattened" block to "next"
 - a pointer to array of 9 "current" blocks, which are essentially neighbours
 - an "out" bool array of 9, where the information about whether there are alive cells on borders (in other words, information whether I need to materialize new blocks)

The kernel is launched with something like this:

```
	const int blockDimension = 64;
	const int threadsPerDimension = 16;
	const dim3 threadsPerBlock(threadsPerDimension, threadsPerDimension);
	const dim3 dimensions(blockDimension / threadsPerBlock.x, blockDimension / threadsPerBlock.y);
	nextGenerationKernel <<< dimensions, threadsPerBlock >>> (next.getDevice(), cudaSurrounding.getDevice(), borderCheck.getDevice());
	auto result = bordersToHost();
```

`bordersToHost()` runs a `cudaMemcpy` in order to get the border information back to host (AFAIK this is horrible because it's synchronous)