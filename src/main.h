#pragma once

#include <string>

#if defined(__CUDACC__) // NVCC
#define alignMem(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define alignMem(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define alignMem(n) __declspec(align(n))
#else
#error "Please provide a definition for alignMem() macro for your host compiler!"
#endif

typedef void* dv_ptr;


void printSep();

extern std::string scenefile;