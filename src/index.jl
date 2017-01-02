# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
@cuda_text """
template<typename T>
__device__ void slowAtomicAdd(T* addr, T val) {}

template<>
__device__ void slowAtomicAdd<float> (float *addr, float val) {
  unsigned int *a = (unsigned int *) addr;
  for (;;) {
    unsigned int old = *a;
    unsigned int swapped = atomicCAS(a, old,
      __float_as_int(__int_as_float(old) + val));
    if (old == swapped) break;
  }
}

template<>
__device__ void slowAtomicAdd<double>(double *addr, double val) {
  unsigned long long *a = (unsigned long long *) addr;
  for (;;) {
    unsigned long long old = *a;
    unsigned long long swapped = atomicCAS(a, old,
      __double_as_longlong(__longlong_as_double(old) + val));
    if (old == swapped) break;
  }
}
"""

### reshape

Base.reshape(a::ANode, i::Int...) = broadcast(a, i)

function Base.reshape(a::ANode, i::Tuple{Vararg{Int}})
  @assert length(a) == prod(i)
  ANode(:reshape, (a,), i)
end

do_forward!(n::ANode{:reshape}, out, in) = copy!(vec(out), vec(in))

gradient_node(n::ANode{:reshape}, wrt::ANode, b::ANode) = reshape(b, size(wrt))

### broadcast

function Base.broadcast(f, a::ANode, b::ANode)
  shape = zeros(Int, max(ndims(a), ndims(b)))
  for i in 1:length(shape)
    if ndims(b) < i || size(b, i) == 1
      shape[i] = size(a, i)
    elseif ndims(a) < i || size(a, i) == 1 || size(b, i) == size(a, i)
      shape[i] = size(b, i)
    else
     error("Incompatible dimensions in broadcast $(size(a)) and $(size(b))")
    end
  end

  return f(broadcast_node(a, (shape...)), broadcast_node(b, (shape...)))
end

broadcast_node(n::ANode, sz::ANode) = broadcast_node(n, size(sz))

function broadcast_node(n::ANode, sz::Tuple{Vararg{Int}})
  if size(n) == sz
    return n
  elseif length(n) == 1
    ANode(:fill, (n,), sz)
  elseif (ndims(n) == length(sz)-1 && all(i->size(n,i) == sz[i], 1:ndims(n))) ||
          (ndims(n) == length(sz) && all(i->size(n,i) == sz[i], 1:ndims(n)-1) && size(n, ndims(n)) == 1)
    ANode(:repeat, (n,), sz)
  elseif (ndims(n) == length(sz)-1 && all(i->size(n,i) == sz[i+1], 1:ndims(n))) ||
          (ndims(n) == length(sz) && all(i->size(n,i) == sz[i], 2:ndims(n)) && size(n, 1) == 1)
    ANode(:repeat_inner, (n,), sz)
  else
    i = ANode(broadcast((x,y)->x, reshape(1:length(n), size(n)), Array(Int, sz)); c=true)
    ANode(:getindex, (n,i), sz)
    error()
  end
end

### fill

_fill{T}(n::Integer, a::T, b::T) = my_fill(Csize_t(n), a, b)

function _fill{T}(n::Csize_t, a::Array{T}, b::Array{T})
  @assert length(a) == n && length(b) == 1
  fill!(a, b[1])
end

@cuda_gsl _fill{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = b[0]"

do_forward!(n::ANode{:fill}, out, in) = _fill(Csize_t(length(out)), out, in)

gradient_node(n::ANode{:fill}, wrt::ANode, b::ANode) = sum(b)

### repeat

function _repeat{T}(n::Csize_t, m::Csize_t, a::Array{T}, b::Array{T})
  @assert length(a) == n && length(b) == m && size(a, ndims(a))*m == n
  k = 0
  for i in 1:size(a, ndims(a))
    a[k + (1:length(b))] = b
    k += length(b)
  end
end

@cuda_gsl _repeat{T<:Real}(n::Csize_t, m::Csize_t, a::&T, b::&T) "a[i] = b[i%m]"

do_forward!(n::ANode{:repeat}, out, in) = _repeat(Csize_t(length(out)), Csize_t(length(in)), out, in)

gradient_node(n::ANode{:repeat}, wrt::ANode, b::ANode) = sum(b, ndims(b))

### repeat_inner

function _repeat_inner{T}(n::Csize_t, m::Csize_t, a::Array{T}, b::Array{T})
  @assert length(a) == n && length(b) == m && size(a, 1)*m == n
  nr = n÷m
  for i in 1:nr
    a[i + (0:nr:nr*length(b)-1)] = b
  end
end

@cuda_gsl _repeat_inner{T<:Real}(n::Csize_t, m::Csize_t, a::&T, b::&T) "a[i] = b[i*m/n]"

do_forward!(n::ANode{:repeat_inner}, out, in) = _repeat_inner(Csize_t(length(out)), Csize_t(length(in)), out, in)

gradient_node(n::ANode{:repeat_inner}, wrt::ANode, b::ANode) = sum(b, 1)

### getindex

function _getindex{T,N}(n::Csize_t, a::Array{T,N}, b::Array{T}, c::Array{T,N})
  @assert length(a) == length(c) == n
  for i in 1:n a[i] = b[round(Int, c[i])] end
end

@cuda_gsl _getindex{T<:Real}(n::Csize_t, a::&T, b::&T, c::&T) "a[i] = b[((int) c[i])-1]"

function bprop_getindex{T,N}(n::Csize_t, a::Array{T}, b::Array{T,N}, c::Array{T,N})
  @assert length(b) == length(c) == n
  for i in 1:n a[round(Int, c[i])] += b[i] end
end

@cuda_gsl bprop_getindex{T<:Real}(n::Csize_t, a::&T, b::&T, c::&T) "slowAtomicAdd(&a[((int) c[i])-1], b[i])"
# // a[c[i]-1] += b[i]; // XXX no atomicAdd for doubles -- how to do this right?"

function Base.getindex{I<:Integer}(n::ANode, i::AbstractArray{I})
  @assert all(1 .<= i .<= length(n))
  ANode(:getindex, (n, ANode(i, c=true)), size(i))
end

do_forward!(n::ANode{:getindex}, out, in1, in2) = _getindex(Csize_t(length(out)), out, in1, in2)

gradient_node(n::ANode{:getindex}, wrt::ANode, b::ANode) = ANode(:getindex_bp, (b,n.input[2]), size(n.input[1]))
function do_forward!(n::ANode{:getindex_bp}, out, in1, in2)
  fill!(out, zero(eltype(out)))
  bprop_getindex(Csize_t(length(in1)), out, in1, in2)
end

### getindex (UnitRange)

function _geturindex{T,N}(n::Csize_t, a::Array{T,N}, b::Array{T}, start::Cint, stop::Cint)
  @assert n == length(a) == stop-start+1
  for i in 1:n a[i] = b[start+i-1] end
end

@cuda_gsl _geturindex{T<:Real}(n::Csize_t, a::&T, b::&T, start::Cint, stop::Cint) "a[i] = b[i+start]"

function bprop_geturindex{T,N}(n::Csize_t, a::Array{T}, b::Array{T,N}, start::Cint, stop::Cint)
  @assert length(b) == n
  for i in 1:n a[i+start-1] += b[i] end
end

@cuda_gsl bprop_geturindex{T<:Real}(n::Csize_t, a::&T, b::&T, start::Cint, stop::Cint) "slowAtomicAdd(&a[start+i], b[i])"
# // a[c[i]-1] += b[i]; // XXX no atomicAdd for doubles -- how to do this right?"

function Base.getindex(n::ANode, i::UnitRange)
  @assert all(1 .<= i .<= length(n))
  ANode(:geturindex, (n,), size(i), i)
end

do_forward!(n::ANode{:geturindex}, out, in1) = _geturindex(Csize_t(length(out)), out, in1, Cint(first(n.arg)), Cint(last(n.arg)))

gradient_node(n::ANode{:geturindex}, wrt::ANode, b::ANode) = ANode(:geturindex_bp, (b,), size(n.input[1]), n.arg)
function do_forward!(n::ANode{:geturindex_bp}, out, in1)
  fill!(out, zero(eltype(out)))
  bprop_geturindex(Csize_t(length(in1)), out, in1, Cint(first(n.arg)), Cint(last(n.arg)))
end
