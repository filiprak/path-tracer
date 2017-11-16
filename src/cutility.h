#pragma once
#include "cuda_runtime.h"

__host__ __device__
inline unsigned int utilhash(unsigned int);

__host__ __device__
inline unsigned int wang_hash(unsigned int);