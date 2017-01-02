# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
@cuda_include "cudaReduce.cu"

### Constructors

for f in [:sum, :minimum, :maximum]
  qf = QuoteNode(f)
  qf_i = QuoteNode(Symbol(f, "_i"))
  _priname = Symbol("_", f, "!")
  @eval begin
    Base.$f(x::ANode) = ANode($qf, (x,), (1,))

    function Base.$f(x::ANode, i::Integer)
      nx = size(x, i)
      ny = prod(Int[size(x,j) for j in 1:(i-1)])
      nz = prod(Int[size(x,j) for j in (i+1):ndims(x)])
      incx = ny
      incy = i == 1 ? size(x, 1): 1
      incz = ny*nx

      sz = [size(x)...]
      sz[i] = 1

      ANode($qf_i, (x,), (sz...), (nx, ny, nz, incx, incy, incz))
    end

    do_forward!(n::ANode{$qf}, out, in) = $_priname(length(in), out, in)
    do_forward!(n::ANode{$qf_i}, out, in) = $_priname(
        n.arg[1], n.arg[2], n.arg[3], out, in, n.arg[4], n.arg[5], n.arg[6])
  end
end

### sum

function _sum!{T}(n::Integer, y::Array{T}, x::Array{T}, a::Real=zero(T))
  @assert n == length(x) && 1 == length(y)
  y[1] = a == 0 ? 0 : y[1]*a
  for i in 1:n @inbounds y[1] += x[i] end
  nothing
end

function _sum!{T}(nx::Integer, ny::Integer, nz::Integer, y::Array{T}, x::Array{T}, incx::Integer, incy::Integer, incz::Integer, a::Real=zero(T))
  @assert ny*nz == length(y)

  for i in 1:nz
    for j in 1:ny
      @inbounds yi = j + ny*(i-1)
      xi = (i-1)*incz + (j-1)*incy + 1

      s = a == 0 ? zero(eltype(y)) : convert(eltype(y), y[yi]*a)
      for k in 1:nx
        s += x[xi]
        xi += incx
      end

      @inbounds y[yi] = s
    end
  end
end

@cuda begin
_sum!{T}(n::Integer, y::CudaArray{T}, x::CudaArray{T}, a::Real=zero(T)) = _sum!(n, 1, 1, y, x, 1, 1, 1, a)

const cuda_sum_32_float = CuFunction(md, "cuda_sum_32_float")
_sum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float32}, x::CudaArray{Float32}, incx::Integer, incy::Integer, incz::Integer, a::Real=zero(Float32)) =
    cudacall(cuda_sum_32_float, (1,ny,nz), 32, (Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, Coff_t, Coff_t, Coff_t), nx, x, y, a, incx, incy, incz; shmem_bytes=32*4, stream=STREAM[1])

const cuda_sum_32_double = CuFunction(md, "cuda_sum_32_double")
_sum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float64}, x::CudaArray{Float64}, incx::Integer, incy::Integer, incz::Integer, a::Real=zero(Float64)) =
    cudacall(cuda_sum_32_double, (1,ny,nz), 32, (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Coff_t, Coff_t, Coff_t), nx, x, y, a, incx, incy, incz; shmem_bytes=32*8, stream=STREAM[1])
end

function add_constant{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert n == length(a) && 1 == length(b)
    a[:] += b[1]
end

@cuda_gsl add_constant{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] += b[0]"

gradient_node(n::ANode{:sum}, wrt::ANode, b::ANode) = broadcast_node(b, size(wrt))
gradient_node(n::ANode{:sum_i}, wrt::ANode, b::ANode) = broadcast_node(b, size(wrt))

### minimum

function _minimum!{T}(n::Integer, y::Array{T}, x::Array{T})
  @assert n == length(x) && 1 == length(y)
  y[1] = x[1]
  for i in 2:n
    @inbounds if x[i] < y[1] y[1] = x[i] end
  end
  nothing
end

function _minimum!{T}(nx::Integer, ny::Integer, nz::Integer, y::Array{T}, x::Array{T}, incx::Integer, incy::Integer, incz::Integer)
  @assert ny*nz == length(y)

  for i in 1:nz
    for j in 1:ny
      @inbounds yi = j + ny*(i-1)
      xi = (i-1)*incz + (j-1)*incy + 1

      s = x[xi]
      xi += incx
      for k in 2:nx
        if x[xi] < s s = x[xi] end
        xi += incx
      end

      @inbounds y[yi] = s
    end
  end
end

@cuda begin
_minimum!{T}(n::Integer, y::CudaArray{T}, x::CudaArray{T}) = _minimum!(n, 1, 1, y, x, 1, 1, 1)

const cuda_minimum_32_float = CuFunction(md, "cuda_minimum_32_float")
_minimum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float32}, x::CudaArray{Float32}, incx::Integer, incy::Integer, incz::Integer) =
    cudacall(cuda_minimum_32_float, (1,ny,nz), 32, (Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Coff_t, Coff_t, Coff_t), nx, x, y, incx, incy, incz; shmem_bytes=32*4, stream=STREAM[1])

const cuda_minimum_32_double = CuFunction(md, "cuda_minimum_32_double")
_minimum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float64}, x::CudaArray{Float64}, incx::Integer, incy::Integer, incz::Integer) =
    cudacall(cuda_minimum_32_double, (1,ny,nz), 32, (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Coff_t, Coff_t, Coff_t), nx, x, y, incx, incy, incz; shmem_bytes=32*8, stream=STREAM[1])
end

gradient_node(n::ANode{:minimum}, wrt::ANode, b::ANode) = b.*(n .== n.input[1])
gradient_node(n::ANode{:minimum_i}, wrt::ANode, b::ANode) = b.*(n .== n.input[1])

### maximum

function _maximum!{T}(n::Integer, y::Array{T}, x::Array{T})
  @assert n == length(x) && 1 == length(y)
  y[1] = x[1]
  for i in 2:n
    @inbounds if x[i] > y[1] y[1] = x[i] end
  end
  nothing
end

function _maximum!{T}(nx::Integer, ny::Integer, nz::Integer, y::Array{T}, x::Array{T}, incx::Integer, incy::Integer, incz::Integer)
  @assert ny*nz == length(y)

  for i in 1:nz
    for j in 1:ny
      @inbounds yi = j + ny*(i-1)
      xi = (i-1)*incz + (j-1)*incy + 1

      s = x[xi]
      xi += incx
      for k in 2:nx
        if x[xi] > s s = x[xi] end
        xi += incx
      end

      @inbounds y[yi] = s
    end
  end
end

@cuda begin
_maximum!{T}(n::Integer, y::CudaArray{T}, x::CudaArray{T}) = _maximum!(n, 1, 1, y, x, 1, 1, 1)

const cuda_maximum_32_float = CuFunction(md, "cuda_maximum_32_float")
_maximum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float32}, x::CudaArray{Float32}, incx::Integer, incy::Integer, incz::Integer) =
    cudacall(cuda_maximum_32_float, (1,ny,nz), 32, (Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Coff_t, Coff_t, Coff_t), nx, x, y, incx, incy, incz; shmem_bytes=32*4, stream=STREAM[1])

const cuda_maximum_32_double = CuFunction(md, "cuda_maximum_32_double")
_maximum!(nx::Integer, ny::Integer, nz::Integer, y::CudaArray{Float64}, x::CudaArray{Float64}, incx::Integer, incy::Integer, incz::Integer) =
    cudacall(cuda_maximum_32_double, (1,ny,nz), 32, (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Coff_t, Coff_t, Coff_t), nx, x, y, incx, incy, incz; shmem_bytes=32*8, stream=STREAM[1])
end

gradient_node(n::ANode{:maximum}, wrt::ANode, b::ANode) = b.*(n .== n.input[1])
gradient_node(n::ANode{:maximum_i}, wrt::ANode, b::ANode) = b.*(n .== n.input[1])
