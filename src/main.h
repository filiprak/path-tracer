#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "kernel.h"

#if defined(__CUDACC__) // NVCC
#define alignMem(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define alignMem(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define alignMem(n) __declspec(align(n))
#else
#error "Please provide a definition for alignMem() macro for your host compiler!"
#endif

void printSep();

extern std::string scenefile;