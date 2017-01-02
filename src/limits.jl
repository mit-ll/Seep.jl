# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
# min

Base.min(v::Real, n::ANode) = min(n, v)
Base.min(n::ANode, v::Real) = ANode(:min_c, (n,), size(n), v)

function Base.min(a::ANode, b::ANode)
  @assert size(a) == size(b)
  ANode(:min, (a,b), size(a))
end

function _min{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::T)
  @assert length(a) == length(y) == n
  @simd for i in 1:n
    @inbounds y[i] = min(a[i], b)
  end
end

function _min{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::Array{T,N})
  @assert length(a) == length(b) == length(y) == n
  @simd for i in 1:n
    @inbounds y[i] = min(a[i], b[i])
  end
end

@cuda_gsl _min{T<:Real}(n::Csize_t, y::&T, a::&T, b::T) "y[i] = min(a[i], b)"
@cuda_gsl _min{T<:Real}(n::Csize_t, y::&T, a::&T, b::&T) "y[i] = min(a[i], b[i])"

do_forward!(n::ANode{:min_c}, y, a) = _min(Csize_t(length(y)), y, a, convert(eltype(y), n.arg))
do_forward!(n::ANode{:min}, y, a, b) = _min(Csize_t(length(y)), y, a, b)

gradient_node(n::ANode{:min_c}, wrt::ANode, b::ANode) = b.*(n .!= n.arg)
gradient_node(n::ANode{:min}, wrt::ANode, b::ANode) = b.*(n .== wrt)

# max
Base.max(v::Real, n::ANode) = max(n, v)
Base.max(n::ANode, v::Real) = ANode(:max_c, (n,), size(n), v)

function Base.max(a::ANode, b::ANode)
  @assert size(a) == size(b)
  ANode(:max, (a,b), size(a))
end

function _max{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::T)
  @assert length(a) == length(y) == n
  @simd for i in 1:n
    @inbounds y[i] = max(a[i], b)
  end
end

function _max{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::Array{T,N})
  @assert length(a) == length(b) == length(y) == n
  @simd for i in 1:n
    @inbounds y[i] = max(a[i], b[i])
  end
end

@cuda_gsl _max{T<:Real}(n::Csize_t, y::&T, a::&T, b::T) "y[i] = max(a[i], b)"
@cuda_gsl _max{T<:Real}(n::Csize_t, y::&T, a::&T, b::&T) "y[i] = max(a[i], b[i])"

do_forward!(n::ANode{:max_c}, y, a) = _max(Csize_t(length(y)), y, a, convert(eltype(y), n.arg))
do_forward!(n::ANode{:max}, y, a, b) = _max(Csize_t(length(y)), y, a, b)

gradient_node(n::ANode{:max_c}, wrt::ANode, b::ANode) = b.*(n .!= n.arg)
gradient_node(n::ANode{:max}, wrt::ANode, b::ANode) = b.*(n .== wrt)

# min (const)

mutates(::ANode{:min!}) = true

min!(v::Real, n::ANode) = min!(n, v)
min!(n::ANode, v::Real) = ANode(:min!, (n,), size(n), v)

function _min{T}(n::Csize_t, x::Array{T}, v::T)
  @assert length(x) == n
  for i in 1:n
    @inbounds if x[i] > v x[i] = v end
  end
end

@cuda_gsl _min{T<:Real}(n::Csize_t, x::&T, v::T) "if (x[i] > v) x[i] = v"

do_forward!(n::ANode{:min!}, x) = _min(Csize_t(length(x)), x, convert(eltype(x), arg(n)))
gradient_node(n::ANode{:min!}, wrt::ANode, b::ANode) = b.*(n .!= n.arg)

# max (const)

mutates(::ANode{:max!}) = true

max!(v::Real, n::ANode) = max!(n, v)
max!(n::ANode, v::Real) = ANode(:max!, (n,), size(n), v)

function _max{T}(n::Csize_t, x::Array{T}, v::T)
  @assert length(x) == n
  for i in 1:n
    @inbounds if x[i] < v x[i] = v end
  end
end

@cuda_gsl _max{T<:Real}(n::Csize_t, x::&T, v::T) "if (x[i] < v) x[i] = v"

do_forward!(n::ANode{:max!}, x) = _max(Csize_t(length(x)), x, convert(eltype(x), arg(n)))
gradient_node(n::ANode{:max!}, wrt::ANode, b::ANode) = b.*(n .!= n.arg)

# clip

mutates(::ANode{:clip!}) = true

function clip{T}(n::Csize_t, x::Array{T}, v::T)
  @assert length(x) == n
  for i in 1:n
    @inbounds if abs(x[i]) > v x[i] = copysign(v, x[i]) end
  end
end

@cuda_gsl clip{T<:Real}(n::Csize_t, x::&T, v::T) "if (fabs(x[i]) > v) x[i] = copysignf(v, x[i])"

clip!(n::ANode, v) = ANode(:clip!, (x,), size(x), v)
do_forward!(n::ANode{:clip!}, x) = clip(x, convert(eltype(x), arg(n)))
