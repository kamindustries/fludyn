#ifndef FLUDYN_COMMON_CUH
#define FLUDYN_COMMON_CUH

#pragma once

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include "allocore/al_Allocore.hpp"  //why cant i do this?

#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

extern dim3 grid, threads;

extern int dimX, dimY, size;
extern int win_x, win_y;
extern int numVertices;
extern int internalFormat;

extern float dt;
extern float diff;
extern float visc;
extern float force;
extern float buoy;
extern float source_density;
extern float dA; // diffusion constants
extern float dB;
extern const char *outputImagePath;

extern float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
extern float *vel[2], *vel_prev[2];
extern float *pressure, *pressure_prev;
extern float *temperature, *temperature_prev;
extern float *density, *density_prev;
extern float *divergence;
extern int *boundary;

extern float4 *displayPtr, *fboPtr, *displayPtr_d;
extern float2 *displayVertPtr;

#endif
