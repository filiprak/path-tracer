# pragma once

#define  cuassert( X )	if ( !(X) ) \
						printf("\n>  CUDA ASSERT FAIL: %s:%d\n\n",  __FILE__, __LINE__);

#define  cudaOk( X )	if ( !(X == cudaSuccess) ) \
						printf("\n>  CUDA RESULT FAIL: %s:%d\n\n",  __FILE__, __LINE__);

int printCudaDevicesInfo();
bool checkCudaError(const char*);