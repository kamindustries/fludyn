#ifndef FLUDYN_CUDAFUNC_CUH
#define FLUDYN_CUDAFUNC_CUH

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "common.cuh"
#include "kernels.cuh"

//TODO: include $CUDA_HOME/samples/common/inc/
//#include <helper_cuda.h>         // helper functions for CUDA error check

#include <cuda.h>


void initCUDA(cudaGraphicsResource_t cgrTx, GLuint txBuffer);

void initGPUArrays();

void reset();

void getMappedPointer(float4 *data, cudaGraphicsResource_t cudaGraphRsrc);

void drawSquare(float *field, float value);

void makeColor(float *data, float4 *toDisplay);

#endif
