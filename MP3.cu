
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

#define size 125
#define f_size sizeof(float)

// multiply using GPU: 1x product calculated per thread
__global__ void multiply_gpu(float *M, float *N, float *P)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < size && column < size)
	{
		float sum = 0;
		for (int i = 0; i < size; ++i) {
			sum += M[row * size + i] * N[i * size + column];
		}

		P[row * size + column] = sum;
	}
}


int main()
{
	cudaEvent_t start, stop;
	float result;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *M = (float *)malloc(size * size * f_size);
	float *N = (float *)malloc(size * size * f_size);
	float *P = (float *)malloc(size * size * f_size);

	float *device_M;
	float *device_N;
	float *device_P;
	cudaMalloc((void **)&device_M, size * size * f_size);
	cudaMalloc((void **)&device_N, size * size * f_size);
	cudaMalloc((void **)&device_P, size * size * f_size);

	// GPU MUL --------------------------------------------------------
	// similar to MP2
	srand(time(NULL));
	for (int i = 0; i < size*size; i++)
	{
		M[i] = ((float)rand() / RAND_MAX) * (1000);
		N[i] = ((float)rand() / RAND_MAX) * (1000);
	}

	// print out transfer time host->device
	cudaEventRecord(start, 0);
	cudaMemcpy(device_M, M, size * size * f_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_N, N, size * size * f_size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("%d \t host->device transfer time: %.2f\n", size, result);

	// 1 thread for per product calculation
	dim3 blockSize(16, 16);
	dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

	cudaEventRecord(start, 0);
	multiply_gpu << <gridSize, blockSize >> >(device_M, device_N, device_P);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);

	printf("GPU RESULT: %.2f\n", result);

	cudaMemcpy(P, device_P, size * size * f_size, cudaMemcpyDeviceToHost);

	// print out transfer time device->host
	cudaEventRecord(start, 0);
	cudaMemcpy(M, device_M, size * size * f_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(N, device_N, size * size * f_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("%d \t device->host transfer time: %.2f\n", size, result);

	bool passed = true;
	int row, col;
	float sum;

	// verification, but also calculate CPU compute time (if-statement time negligible)
	cudaEventRecord(start, 0);
	for (int i = 0; i < size * size; i++)
	{
		row = i / size;
		col = i % size;
		sum = 0;

		// calculate product using CPU to verify GPU output
		for (int k = 0; k < size; k++) {
			sum += M[row * size + k] * N[k * size + col];
		}

		// tolerance of 0.1
		if (fabs(P[i] - sum) > 0.1)
		{
			passed = false;
			break;
		}
	}

	if (passed)
	{
		printf("Test PASSED\n");
	}
	else
	{
		printf("Test FAILED\n");
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("%d \t CPU Time: %.2f\n", size, result);

	free(M);
	free(N);
	free(P);
	cudaFree(device_M);
	cudaFree(device_N);
	cudaFree(device_P);
}