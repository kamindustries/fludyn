#ifndef FLUDYN_CUDAFUNC_CUH
#define FLUDYN_CUDAFUNC_CUH

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>         // helper functions for CUDA error check
#include "common.hpp"
#include "kernels.cuh"

//TODO: example showing simple but cool kernel work. just use cudaMemcpy

#include <cuda.h>


// void initCUDA(cudaGraphicsResource_t cgrTx, GLuint txBuffer); // couldnt get to work

void initGPUArrays();

void reset();

// void getMappedPointer(float4 *data, cudaGraphicsResource_t cudaGraphRsrc); // couldnt get to work

void drawSquare(float *field, float value);

void makeColor(float *data, float4 *toDisplay);

void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
                  float *u, float *v, int *bounds, float dt );

#endif
