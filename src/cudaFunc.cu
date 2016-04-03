#include "cudaFunc.cuh"

// void initCUDA(cudaGraphicsResource_t cgrTx, GLuint txBuffer) {
//   checkCudaErrors( cudaSetDevice(gpuGetMaxGflopsDeviceId()) );
//   checkCudaErrors( cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()) );
//   checkCudaErrors( cudaGraphicsGLRegisterBuffer(&cgrTx, txBuffer, cudaGraphicsMapFlagsWriteDiscard) );
//   cudaGraphicsGLRegisterBuffer( &cgrVertData, vertexArrayID, cudaGraphicsMapFlagsWriteDiscard );
// }

void initGPUArrays() {
  for (int i=0; i<2; i++){
		cudaMalloc((void**)&vel[i], sizeof(int)*size);
		cudaMalloc((void**)&vel_prev[i], sizeof(int)*size);
	}

  cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&chemB, sizeof(float)*size);
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);
	cudaMalloc((void**)&boundary, sizeof(int)*size);

	cudaMalloc((void**)&pressure, sizeof(float)*size );
	cudaMalloc((void**)&pressure_prev, sizeof(float)*size );
	cudaMalloc((void**)&temperature, sizeof(float)*size );
	cudaMalloc((void**)&temperature_prev, sizeof(float)*size );
	cudaMalloc((void**)&density, sizeof(float)*size );
	cudaMalloc((void**)&density_prev, sizeof(float)*size );
	cudaMalloc((void**)&divergence, sizeof(float)*size );

  cudaMalloc((void**)&displayPtr_d, sizeof(float4)*size );

  for (int i=0; i<2; i++){
	  ClearArray<<<grid,threads>>>(vel[i], 0.0, dimX, dimY);
	  ClearArray<<<grid,threads>>>(vel_prev[i], 0.0, dimX, dimY);
  }

  ClearArray<<<grid,threads>>>(chemA, 1.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemA_prev, 1.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemB, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(chemB_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(boundary, 0.0, dimX, dimY);

  ClearArray<<<grid,threads>>>(pressure, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(temperature, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(temperature_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(density, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(density_prev, 0.0, dimX, dimY);
  ClearArray<<<grid,threads>>>(divergence, 0.0, dimX, dimY);

  printf("initGPUArrays(): Initialized GPU arrays.\n");
}

void reset() {
  initGPUArrays();
}

// void getMappedPointer(float4 *data, cudaGraphicsResource_t cudaGraphRsrc){
  // size_t  sizeT;
  // cudaGraphicsMapResources( 1, &cudaGraphRsrc, 0 );
  // cudaGraphicsResourceGetMappedPointer((void**)&data, &sizeT, cudaGraphRsrc);
  // cudaGraphicsUnmapResources( 1, &cudaGraphRsrc, 0 );
// }

void drawSquare(float *field, float value) {
  // ClearArray<<<grid,threads>>>(field, 1.0, dimX, dimY);
  DrawSquare<<<grid,threads>>>(field, value, dimX, dimY);
}

void makeColor(float *data, float4 *toDisplay){
  MakeColor<<<grid,threads>>>(data, toDisplay, dimX, dimY);
}

void dens_step (  float *_chemA, float *_chemA0, float *_chemB, float *_chemB0,
                  float *u, float *v, int *bounds, float dt )
{
  // Naive ARD-----------------------
	AddSource<<<grid,threads>>>(_chemB, _chemB0, dt, dimX, dimY);
	_chemA0 = _chemA;
	_chemB0 = _chemB;
	for (int i = 0; i < 10; i++){
		Diffusion<<<grid,threads>>>(_chemA, laplacian, bounds, dA, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemA, laplacian, dimX, dimY);
    SetBoundary<<<grid,threads>>>(0, _chemA, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		Diffusion<<<grid,threads>>>(_chemB, laplacian, bounds, dB, dt, dimX, dimY);
		AddLaplacian<<<grid,threads>>>(_chemB, laplacian, dimX, dimY);
    SetBoundary<<<grid,threads>>>(0, chemB, bounds, dimX, dimY);
		ClearArray<<<grid,threads>>>(laplacian, 0.0, dimX, dimY);

		// for (int j = 0; j < 1; j++){
    React<<<grid,threads>>>( _chemA, _chemB, bounds, dt, dimX, dimY );
    // }
	}

	// SWAP ( _chemA0, _chemA );
	// SWAP ( _chemB0, _chemB );
  //
	// // Density advection: chemB
	// Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], _chemA0, bounds, _chemA,
	// 						dt, 1.0, true, dimX, dimY);
  //
	// // Density advection: chemB
	// Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], _chemB0, bounds, _chemB,
	// 						dt, 1.0, true, dimX, dimY);
}
