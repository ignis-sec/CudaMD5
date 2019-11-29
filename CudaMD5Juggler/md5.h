#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdint.h>



__global__ void getNext(int* iter, uint8_t* result, uint32_t* hash, uint8_t* solbuf, uint32_t* solhash);
__device__ static const char allowed_characters[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
__device__ static const int alphabet_length = 62;
__device__ static const char salt[] = "ignisET1Y";
__device__ static const int saltlen = 9;
__device__ static const int MAX_UNHASHED_LEN = 32;
#define BLOCKSIZE 512
__device__ void CudaMD5(unsigned char* data, int length, uint32_t* a1, uint32_t* b1, uint32_t* c1, uint32_t* d1);
