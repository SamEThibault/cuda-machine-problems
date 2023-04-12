
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#define N 125
#define f_size sizeof(float)

// each thread producing one output matrix element
__global__ void AddElement(float *C, float *A, float *B)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * N + column;

	if (row < N && column < N)
	{
		C[index] = A[index] + B[index];
	}
}

// each thread producing one output matrix row
__global__ void AddRow(float *C, float *A, float *B)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < N)
	{
		int index = row * N;
		for (int i = 0; i < N; i++)
		{
			C[index + i] = A[index + i] + B[index + i];
		}
	}
}

// each thread producing one output matrix column
__global__ void addColumn(float *C, float *A, float *B)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < N)
	{
		for (int i = 0; i < N; i++)
		{
			C[i * N + column] = A[i * N + column] + B[i * N + column];
		}
	}
}

int main()
{
	cudaEvent_t start, stop;
	float result;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *A = (float *)malloc(N * N * f_size);
	float *B = (float *)malloc(N * N * f_size);
	float *C = (float *)malloc(N * N * f_size);

	float *device_A;
	float *device_B;
	float *device_C;
	cudaMalloc((void **)&device_A, N * N * f_size);
	cudaMalloc((void **)&device_B, N * N * f_size);
	cudaMalloc((void **)&device_C, N * N * f_size);

	// TEST 1 --------------------------------------------------------
	srand(time(NULL));
	for (int i = 0; i < N*N; i++)
	{
		A[i] = ((float)rand() / RAND_MAX) * (1000);
		B[i] = ((float)rand() / RAND_MAX) * (1000);
	}

	cudaMemcpy(device_A, A, N * N * f_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, B, N * N * f_size, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	cudaEventRecord(start, 0);
	AddElement << <gridSize, blockSize >> >(device_C, device_A, device_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("TEST 1 RESULT (Add by element): %.2f\n", result);

	cudaMemcpy(C, device_C, N * N * f_size, cudaMemcpyDeviceToHost);

	bool passed = true;
	for (int i = 0; i < N * N; i++)
	{
		// debugging sum output
		//printf("Element Number: %d \t A: %.2f \t B: %.2f \t answer: %.2f\n", i, A[i], B[i], C[i]);

		if (fabs(C[i] - (A[i] + B[i])) > 0.1)
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


	printf("TEST 1 COMPLETE -------------------------\n");

	// TEST 2 --------------------------------------------------------
	srand(time(NULL));
	for (int i = 0; i < N*N; i++)
	{
		A[i] = ((float)rand() / RAND_MAX) * (1000);
		B[i] = ((float)rand() / RAND_MAX) * (1000);
		C[i] = 0;
	}

	cudaMemcpy(device_A, A, N * N * f_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, B, N * N * f_size, cudaMemcpyHostToDevice);

	dim3 gridSizeRow((N + 15) / 16, (N + 15) / 16);
	dim3 blockSizeRow(16, 16);

	cudaEventRecord(start, 0);
	AddRow << <gridSizeRow, blockSizeRow >> >(device_C, device_A, device_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("TEST 2 RESULT (Add by row): %.2f\n", result);

	cudaMemcpy(C, device_C, N * N * f_size, cudaMemcpyDeviceToHost);

	passed = true;
	for (int i = 0; i < N * N; i++)
	{
		// debugging sum output
		//printf("Element Number: %d \t A: %.2f \t B: %.2f \t answer: %.2f\n", i, A[i], B[i], C[i]);

		if (fabs(C[i] - (A[i] + B[i])) > 0.1)
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

	printf("TEST 2 COMPLETE -------------------------\n");

	// TEST 3  --------------------------------------------------------
	srand(time(NULL));
	for (int i = 0; i < N*N; i++)
	{
		A[i] = ((float)rand() / RAND_MAX) * (1000);
		B[i] = ((float)rand() / RAND_MAX) * (1000);
		C[i] = 0;
	}

	cudaMemcpy(device_A, A, N * N * f_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, B, N * N * f_size, cudaMemcpyHostToDevice);

	dim3 gridSizeCol((N + 15) / 16, (N + 15) / 16);
	dim3 blockSizeCol(16, 16);

	cudaEventRecord(start, 0);
	addColumn << <gridSizeCol, blockSizeCol >> >(device_C, device_A, device_B);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("TEST 3 RESULT (Add my col): %.2f\n", result);

	cudaMemcpy(C, device_C, N * N * f_size, cudaMemcpyDeviceToHost);

	passed = true;
	for (int i = 0; i < N * N; i++)
	{
		// debugging sum output
		//printf("Element Number: %d \t A: %.2f \t B: %.2f \t answer: %.2f\n", i, A[i], B[i], C[i]);

		if (fabs(C[i] - (A[i] + B[i])) > 0.1)
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

	printf("TEST 3 COMPLETE -------------------------\n");

	// CPU TEST -------------------------------------
	for (int i = 0; i < N*N; i++)
	{
		A[i] = ((float)rand() / RAND_MAX) * (1000);
		B[i] = ((float)rand() / RAND_MAX) * (1000);
		C[i] = 0;
	}

	// start timer here
	cudaEventRecord(start, 0);
	for (int i = 0; i < N*N; i++)
	{
		C[i] = A[i] + B[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&result, start, stop);
	printf("CPU RESULT: %.2f\n", result);

	cudaDeviceSynchronize();

	free(A);
	free(B);
	free(C);
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);
}

