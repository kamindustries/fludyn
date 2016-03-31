#include "cudaFunc.cuh"

void initCUDA(cudaGraphicsResource_t cgrTx, GLuint txBuffer) {
  // checkCudaErrors( cudaSetDevice(gpuGetMaxGflopsDeviceId()) );
  // checkCudaErrors( cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()) );
  cudaGraphicsGLRegisterBuffer( &cgrTx, txBuffer, cudaGraphicsMapFlagsWriteDiscard );
  // cudaGraphicsGLRegisterBuffer( &cgrVertData, vertexArrayID, cudaGraphicsMapFlagsWriteDiscard );

  cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&chemB, sizeof(float)*size);
	cudaMalloc((void**)&chemB_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);
	cudaMalloc((void**)&boundary, sizeof(int)*size);

	for (int i=0; i<2; i++){
		cudaMalloc((void**)&vel[i], sizeof(int)*size);
		cudaMalloc((void**)&vel_prev[i], sizeof(int)*size);
	}

	cudaMalloc((void**)&pressure, sizeof(float)*size );
	cudaMalloc((void**)&pressure_prev, sizeof(float)*size );
	cudaMalloc((void**)&temperature, sizeof(float)*size );
	cudaMalloc((void**)&temperature_prev, sizeof(float)*size );
	cudaMalloc((void**)&density, sizeof(float)*size );
	cudaMalloc((void**)&density_prev, sizeof(float)*size );
	cudaMalloc((void**)&divergence, sizeof(float)*size );
}

void initGPUArrays() {
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

void getMappedPointer(float4 *data, cudaGraphicsResource_t cudaGraphRsrc){
  size_t  sizeT;
  cudaGraphicsMapResources( 1, &cudaGraphRsrc, 0 );
  cudaGraphicsResourceGetMappedPointer((void**)&data, &sizeT, cudaGraphRsrc);
  cudaGraphicsUnmapResources( 1, &cudaGraphRsrc, 0 );

}

void drawSquare(float *field, float value) {
  // ClearArray<<<grid,threads>>>(field, 1.0, dimX, dimY);
  DrawSquare<<<grid,threads>>>(field, value, dimX, dimY);
}

void makeColor(float *data, float4 *toDisplay){
  MakeColor<<<grid,threads>>>(data, toDisplay, dimX, dimY);
}
