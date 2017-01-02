// Copyright 2016 Massachusetts Institute of Technology. See LICENSE file for details.
// http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf

template <unsigned int blockSize, typename T, typename R>
__device__ void cuda_reduce(R reduce, size_t n, T *g_idata, T *g_odata, off_t incx, off_t incy, off_t incz, double a = 0) {
    extern __shared__ char vdata[];
    T* sdata = (T*) vdata;

    off_t grid_index = blockIdx.y*incy + blockIdx.z*incz;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = blockSize*gridDim.x;

    if (i < n) {
        sdata[tid] = g_idata[incx*i + grid_index];
        i += gridSize;
    }

    while (i < n) {
        reduce(sdata[tid], g_idata[incx*i + grid_index]);
        i += gridSize;
    }

    __syncthreads();

    if (blockSize >= 512) { if (tid < 256 && tid + 256 < n) { reduce(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128 && tid + 128 < n) { reduce(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64 && tid +  64 < n) { reduce(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >=  64) { if (tid <  32 && tid + 32 < n) { reduce(sdata[tid], sdata[tid +  32]); } }
        if (blockSize >=  32) { if (tid <  16 && tid + 16 < n) { reduce(sdata[tid], sdata[tid +  16]); } }
        if (blockSize >=  16) { if (tid <   8 && tid +  8 < n) { reduce(sdata[tid], sdata[tid +   8]); } }
        if (blockSize >=   8) { if (tid <   4 && tid +  4 < n) { reduce(sdata[tid], sdata[tid +   4]); } }
        if (blockSize >=   4) { if (tid <   2 && tid +  2 < n) { reduce(sdata[tid], sdata[tid +   2]); } }
        if (blockSize >=   2) { if (tid <   1 && tid +  1 < n) { reduce(sdata[tid], sdata[tid +   1]); } }
    }

    if (tid == 0) {
        off_t o = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
        if (a == 0)
	    g_odata[o] = sdata[0];
	else
	    g_odata[o] = a*g_odata[o] + sdata[0];
    }
}

struct Sum { template<typename T> __device__ void operator() (T& acc, const T v) { acc += v; } };
struct Min { template<typename T> __device__ void operator() (T& acc, const T v) { if (v < acc) acc = v; } };
struct Max { template<typename T> __device__ void operator() (T& acc, const T v) { if (v > acc) acc = v; } };

extern "C" {
// Sum
  __global__ void cuda_sum_512_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_256_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_128_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_64_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_32_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_16_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_8_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_4_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_2_double(size_t n, double* g_idata, double* g_odata, double a, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }

  __global__ void cuda_sum_512_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_256_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_128_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_64_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_32_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_16_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_8_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_4_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }
  __global__ void cuda_sum_2_float(size_t n, float* g_idata, float* g_odata, float a, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Sum(), n, g_idata, g_odata, incx, incy, incz, a); }

// Max
  __global__ void cuda_maximum_512_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_256_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_128_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_64_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_32_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_16_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_8_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_4_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_2_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Max(), n, g_idata, g_odata, incx, incy, incz); }

  __global__ void cuda_maximum_512_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_256_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_128_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_64_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_32_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_16_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_8_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_4_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Max(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_maximum_2_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Max(), n, g_idata, g_odata, incx, incy, incz); }

// Min
  __global__ void cuda_minimum_512_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_256_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_128_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_64_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_32_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_16_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_8_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_4_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_2_double(size_t n, double* g_idata, double* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Min(), n, g_idata, g_odata, incx, incy, incz); }

  __global__ void cuda_minimum_512_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<512>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_256_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<256>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_128_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<128>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_64_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<64>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_32_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<32>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_16_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<16>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_8_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<8>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_4_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<4>(Min(), n, g_idata, g_odata, incx, incy, incz); }
  __global__ void cuda_minimum_2_float(size_t n, float* g_idata, float* g_odata, off_t incx, off_t incy, off_t incz) { cuda_reduce<2>(Min(), n, g_idata, g_odata, incx, incy, incz); }
}
