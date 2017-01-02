// Copyright 2016 Massachusetts Institute of Technology. See LICENSE file for details.
#include <cstddef>

typedef size_t jl_int;

// Mirrors julia structure
template<int N>
struct ConvOpt {
  jl_int xsize[N];
  jl_int fsize[N];
  jl_int stride[N];
  jl_int padding[N];
};

// Index3
class index3 {
public:
  int x, y, z;
  __device__ index3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
  __device__ index3(int i) : x(i), y(i), z(i) {}
  __device__ index3(dim3 dim) : x(dim.x), y(dim.y), z(dim.z) {}
  __device__ index3(uint3 dim) : x(dim.x), y(dim.y), z(dim.z) {}

  __device__ const index3 operator+(const index3 &o) const { return index3(x+o.x, y+o.y, z+o.z); }
  __device__ const index3 operator-(const index3 &o) const { return index3(x-o.x, y-o.y, z-o.z); }
  __device__ const index3 operator*(const index3 &o) const { return index3(x*o.x, y*o.y, z*o.z); }
  __device__ const index3 operator/(const index3 &o) const { return index3(x/o.x, y/o.y, z/o.z); }

  __device__ bool operator<(const index3 &o) const { return x < o.x && y < o.y && z < o.z; }
  __device__ bool operator>(const index3 &o) const { return x > o.x && y > o.y && z > o.z; }
  __device__ bool operator<=(const index3 &o) const { return x <= o.x && y <= o.y && z <= o.z; }
  __device__ bool operator>=(const index3 &o) const { return x >= o.x && y >= o.y && z >= o.z; }
  __device__ bool operator==(const index3 &o) const { return x == o.x && y == o.y && z == o.z; }
  __device__ bool operator!=(const index3 &o) const { return x != o.x || y != o.y || z != o.z; }

  __device__ int length() const { return x*y*z; }
  __device__ int operator()(const index3 &a) const { return a.x + x*(a.y + y*a.z); }
};

static inline __device__ const index3 get_xsize(const ConvOpt<4> &opt)   { return index3(opt.xsize[0], opt.xsize[1], opt.xsize[2]); }
static inline __device__ const index3 get_fsize(const ConvOpt<4> &opt)   { return index3(opt.fsize[0], opt.fsize[1], opt.fsize[2]); }
static inline __device__ const index3 get_stride(const ConvOpt<4> &opt)  { return index3(opt.stride[0], opt.stride[1], opt.stride[2]); }
static inline __device__ const index3 get_padding(const ConvOpt<4> &opt) { return index3(opt.padding[0], opt.padding[1], opt.padding[2]); }
static inline __device__ const index3 get_ysize(const ConvOpt<4> &opt)   { return (get_xsize(opt)-get_fsize(opt)+get_padding(opt)*2)/get_stride(opt) + 1; }
static inline __device__ const index3 get_offset(const ConvOpt<4> &opt)  { return index3(blockDim)*index3(blockIdx); }

static inline __device__ bool increment(index3 &index, const index3 &limit, const index3& origin) {
    index.x += blockDim.x * gridDim.x;
    if (index.x < limit.x) return true;
    index.x = origin.x;
    index.y += blockDim.y * gridDim.y;
    if (index.x < limit.x) return true;
    index.y = origin.y;
    index.z += blockDim.z * gridDim.z;
    if (index.z < limit.z) return true;
    index.z = origin.z;
    return false;
}

// Array Accessors
template<typename T>
class ArrayAccessor {
  const T *ptr; 
  index3 size;

public:
  __device__ ArrayAccessor(const T *iptr, const index3 &isize): ptr(iptr), size(isize) {}
  __device__ const T operator[](const index3 &i) const { return ptr[size(i)]; }
};

// __ldg (available in compute capability >= 3.5 enables caching values that are
// constant over the life of the kernel.
#if __CUDA_ARCH__ >= 350
template<typename T>
static inline __device__ const T ldg(const T& x) { return __ldg(&x); };
#else
template<typename T>
static inline __device__ const T ldg(const T& x) { return x; };
#endif

template<typename T>
static inline int clamp(T i, T lo, T hi) { return lo < i ? lo : hi > i ? hi : i; }

template<typename T>
class LdgArrayAccessor {
  const T *ptr; 
  index3 size;

public:
  __device__ LdgArrayAccessor(const T *iptr, const index3 &isize): ptr(iptr), size(isize) {}
  __device__ const T operator[](const index3 &i) const { return ldg(ptr[size(i)]); }
};

// Shared memory cache
template<typename T>
class SharedMemoryCacheAccessor {
  T * const ptr;
  const index3 offset;
  const index3 size;

public:
  __device__ SharedMemoryCacheAccessor(T * const smptr, const T *data,
      const index3 &ioffset, const index3 &isize, const index3 &limit):
      ptr(smptr), offset(ioffset), size(isize) {
    int i, j, k;
    for (k = threadIdx.z; k < size.z; k += blockDim.z)
    for (j = threadIdx.y; j < size.y; j += blockDim.y)
    for (i = threadIdx.x; i < size.x; i += blockDim.x) {
      index3 ii(i,j,k);
      if (ii < size && ii+offset < limit) {
        ptr[size(ii)] = data[limit(ii+offset)];
      }
    }
    __syncthreads();
  }
  __device__ const T operator[](const index3 &i) const { return ptr[size(i-offset)]; }
};

template<typename T>
__device__ void conv3(const ConvOpt<4> &opt, const jl_int nf,
	      T * __restrict__ y, const T * __restrict__ f, const T * __restrict__ x) {
  extern __shared__ char sm[];
  const index3 xsize = get_xsize(opt);
  const index3 fsize = get_fsize(opt);
  const index3 stride = get_stride(opt);
  const index3 padding = get_padding(opt);
  const index3 ysize = get_ysize(opt) * index3(nf, 1, 1);
  index3 offset = get_offset(opt);

  int nz = gridDim.z/opt.xsize[3];
  int img = blockIdx.z/nz;
  offset.z = blockDim.z*(blockIdx.z%nz);
  y += ysize.length()*img;
  x += xsize.length()*img;

  //index3 blksize = (index3(blockDim)-1)*stride + fsize;
  //SharedMemoryCacheAccessor<T> xa((T*) sm, x, offset*stride, blksize, xsize);
  ArrayAccessor<T> xa(x, xsize);

  index3 opos = offset + index3(threadIdx);
  index3 opos2 = opos;
  opos2.x /= nf;

  if (opos < ysize) {
    int j1, j2, j3;
    T a = 0;

    for (j3 = 0; j3 < fsize.z; j3++)
    for (j2 = 0; j2 < fsize.y; j2++)
    for (j1 = 0; j1 < fsize.x; j1++) {
      index3 jj(j1,j2,j3);
      a += ldg(f[(opos.x%nf) + nf*fsize(jj)]) * xa[jj + opos2*stride];
    }

    y[ysize(opos)] = a;
  }
}

template<typename T>
__device__ void conv3_bp_filter(const ConvOpt<4> &opt, const jl_int nf,
              T * __restrict__ df, const T * __restrict__ x, const T * __restrict__ b) {
  extern __shared__ char sm[];
  const index3 xsize = get_xsize(opt);
  const index3 fsize = get_fsize(opt);
  const index3 stride = get_stride(opt);
  const index3 padding = get_padding(opt);
  const index3 ysize = get_ysize(opt);
  index3 offset = get_offset(opt);

  //LdgArrayAccessor<T> ba(b, ysize);
  //index3 blksize = (index3(blockDim)-1)*stride + fsize;
  //SharedMemoryCacheAccessor<T> xa((T*) sm, x, offset*stride, blksize, xsize);

  index3 opos = index3(threadIdx) + offset;
  int j0 = opos.x % nf;
  index3 opos2 = opos;
  opos2.x /= nf;

  if (opos2 < fsize) {
    T a = 0;
    int img;
    for (img = 0; img < opt.xsize[3]; img++) {
      int j1, j2, j3;
      for (j3 = 0; j3 < ysize.z; j3++)
      for (j2 = 0; j2 < ysize.y; j2++)
      for (j1 = 0; j1 < ysize.x; j1++) {
        index3 jj(j1,j2,j3);
        index3 ii = jj*stride + opos2;
        if (ii >= 0 && ii < xsize)
          a += ldg(x[xsize(ii) + img*xsize.length()]) * b[j0 + nf*ysize(jj) + img*nf*ysize.length()];
      }
    }

    df[j0 + nf*fsize(opos2)] = a;
  }
}

template<typename T>
__device__ void conv3_bp_data(const ConvOpt<4> &opt, const jl_int nf,
              T * __restrict__ dx, const T * __restrict__ f, const T * __restrict__ b) {
  const index3 xsize = get_xsize(opt);
  const index3 fsize = get_fsize(opt);
  const index3 stride = get_stride(opt);
  const index3 padding = get_padding(opt);
  const index3 ysize = get_ysize(opt);
  index3 offset = get_offset(opt);

  int nz = gridDim.z/opt.xsize[3];
  int img = blockIdx.z/nz;
  offset.z = blockDim.z*(blockIdx.z%nz);
  b += nf*ysize.length()*img;
  dx += xsize.length()*img;

  index3 opos = index3(threadIdx) + offset;

  if (opos < xsize) {
    T a = 0;
    int j0, j1, j2, j3;
    for (j0 = 0; j0 < nf; j0++) {
      for (j3 = 0; j3 < fsize.z; j3++)
      for (j2 = 0; j2 < fsize.y; j2++)
      for (j1 = 0; j1 < fsize.x; j1++) {
        index3 jj(j1,j2,j3);
	index3 ii = opos*stride - jj;
        if (ii >= 0 && ii < ysize)
          a += ldg(f[j0 + nf*fsize(jj)]) * b[j0 + nf*ysize(ii)];
      }
    }
    dx[xsize(opos)] = a;
  }
}

extern "C" {
    __global__ void conv3_Float32(const ConvOpt<4> *opt, jl_int nf,
                  float * __restrict__ y, const float * __restrict__ f, const float * __restrict__ x) {
        conv3<float>(*opt, nf, y, f, x);
    }

    __global__ void conv3_Float64(const ConvOpt<4> *opt, jl_int nf,
                  double * __restrict__ y, const double * __restrict__ f, const double * __restrict__ x) {
        conv3<double>(*opt, nf, y, f, x);
    }

    __global__ void conv3_bp_data_Float32(const ConvOpt<4> *opt, jl_int nf,
                  float * __restrict__ dx, const float * __restrict__ f, const float * __restrict__ b) {
        conv3_bp_data<float>(*opt, nf, dx, f, b);
    }

    __global__ void conv3_bp_data_Float64(const ConvOpt<4> *opt, jl_int nf,
                  double * __restrict__ dx, const double * __restrict__ f, const double * __restrict__ b) {
        conv3_bp_data<double>(*opt, nf, dx, f, b);
    }

    __global__ void conv3_bp_filter_Float32(const ConvOpt<4> *opt, jl_int nf,
                  float * __restrict__ df, const float * __restrict__ x, const float * __restrict__ b) {
        conv3_bp_filter<float>(*opt, nf, df, x, b);
    }

    __global__ void conv3_bp_filter_Float64(const ConvOpt<4> *opt, jl_int nf,
                  double * __restrict__ df, const double * __restrict__ x, const double * __restrict__ b) {
        conv3_bp_filter<double>(*opt, nf, df, x, b);
    }
}

// Pooling

template<typename T>
__device__ void pool(const ConvOpt<4> &opt, T * __restrict__ y, const T * __restrict__ x) {
  extern __shared__ char sm[];
  const index3 xsize = get_xsize(opt);
  const index3 fsize = get_fsize(opt);
  const index3 stride = get_stride(opt);
  const index3 padding = get_padding(opt);
  const index3 ysize = get_ysize(opt);
  index3 offset = get_offset(opt);

  int nz = gridDim.z/opt.xsize[3];
  int img = blockIdx.z/nz;
  offset.z = blockDim.z*(blockIdx.z%nz);
  y += ysize.length()*img;
  x += xsize.length()*img;

  index3 blksize = (index3(blockDim)-1)*stride + fsize;
  SharedMemoryCacheAccessor<T> xa((T*) sm, x, offset*stride, blksize, xsize);

  index3 opos = offset + index3(threadIdx);
  if (opos < ysize) {
    int j1, j2, j3;
    T a = xa[opos*stride];

    for (j3 = 0; j3 < fsize.z; j3++)
    for (j2 = 0; j2 < fsize.y; j2++)
    for (j1 = 0; j1 < fsize.x; j1++) {
      index3 jj(j1,j2,j3);
      index3 ii = jj + opos*stride;
      if (ii < xsize) {
        T b = xa[ii];
        if (b > a) a = b;
      }
    }

    y[ysize(opos)] = a;
  }
}

template<typename T>
__device__ void pool_bp(const ConvOpt<4> &opt, T * __restrict__ dx, const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ b) {
  const index3 xsize = get_xsize(opt);
  const index3 fsize = get_fsize(opt);
  const index3 stride = get_stride(opt);
  const index3 padding = get_padding(opt);
  const index3 ysize = get_ysize(opt);
  index3 offset = get_offset(opt);

  int nz = gridDim.z/opt.xsize[3];
  int img = blockIdx.z/nz;
  offset.z = blockDim.z*(blockIdx.z%nz);
  y += ysize.length()*img;
  b += ysize.length()*img;
  x += xsize.length()*img;
  dx += xsize.length()*img;

  ArrayAccessor<T> xa(x, xsize);
  ArrayAccessor<T> ya(y, ysize);
  ArrayAccessor<T> ba(b, ysize);

  const index3 opos = offset + index3(threadIdx);
  if (opos < ysize) {
    const T zi = ya[opos];

    int j1, j2, j3;
    for (j3 = 0; j3 < fsize.z; j3++)
    for (j2 = 0; j2 < fsize.y; j2++)
    for (j1 = 0; j1 < fsize.x; j1++) {
      index3 jj(j1,j2,j3);
      index3 ii = jj + opos*stride;
      if (ii < xsize && xa[ii] == zi) {
        dx[xsize(ii)] += ba[opos];
      }
    }
  }
}

extern "C" {
    __global__ void pool_Float32(const ConvOpt<4> *opt,
                  float * __restrict__ y, const float * __restrict__ x) {
        pool<float>(*opt, y, x);
    }

    __global__ void pool_Float64(const ConvOpt<4> *opt,
                  double * __restrict__ y, const double * __restrict__ x) {
        pool<double>(*opt, y, x);
    }

    __global__ void pool_bp_Float32(const ConvOpt<4> *opt,
                  float * __restrict__ dx, const float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ b) {
        pool_bp<float>(*opt, dx, x, y, b);
    }

    __global__ void pool_bp_Float64(const ConvOpt<4> *opt,
                  double * __restrict__ dx, const double * __restrict__ x, const double * __restrict__ y, const double * __restrict__ b) {
        pool_bp<double>(*opt, dx, x, y, b);
    }
}
