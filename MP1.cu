// ELEC 374 - Machine Problem 1
#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>

// Find processor core count based on release version (2.1, for example)
int find_core_count(int major, int minor)
{
    int count = 0;

    if (major == 1)
    {
        count = 8;
    }
    else if (major == 2 && minor == 0)
    {
        count = 32;
    }
    else if (major == 2 && minor == 1)
    {
        count = 48;
    }
    else if (major == 3)
    {
        count = 192;
    }
    else if (major == 5)
    {
        count = 128;
    }

    return count;
}

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("Number of devices: %d\n\n", device_count);

    for (int device_num = 0; device_num < device_count; device_num++)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_num);

        printf("Device %d: %s \n\n", device_num, device_prop.name);
        printf("Properties: \n");

        // we need: clock rate, num of streaming multiprocessors, num of cores, warp size, amount of global memory, amount of cst memory,
        // amount of shared memory per block, number of registers available per block, max number of threads per block, max size of each dimension of a block,
        // max size of each dimension of a grid

        printf("Clock rate: %d\n", device_prop.clockRate);
        printf("Number of streaming multiprocessors: %d\n", device_prop.multiProcessorCount);

        // total num cores
        int num_cores = find_core_count(device_prop.major, device_prop.minor) * device_prop.multiProcessorCount;

        if (num_cores == 0)
        {
            printf("Number of cores per multiprocessor: UNKNOWN\n");
        }
        else
        {
            printf("Number of cores: %d\n", num_cores);
        }

        printf("Warp size: %d\n", device_prop.warpSize);
        printf("Amount of global memory: %zu bytes\n", device_prop.totalGlobalMem);
        printf("Amount of constant memory: %zu bytes\n", device_prop.totalConstMem);
        printf("Amount of shared memory per block: %zu bytes\n", device_prop.sharedMemPerBlock);
        printf("Number of registers available per block: %d\n", device_prop.regsPerBlock);
        printf("Max number of threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Max size of each dimension of a block: %d x %d x %d \n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max size of each dimension of a grid: %d x %d x %d \n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
        printf("\n\n");
    }
}
