# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
mutates(::ANode{:store!}) = true

arg(x::ANode{:axpy!}) = arg(x.input[1])
arg(x::ANode{:store!}) = arg(x.input[1])

function store!(x::ANode, y::ANode)
  @assert size(x) == size(y)
  return ANode(:store!, (x, y), size(x))
end

function do_forward!(n::ANode{:store!}, y, x)
  @assert length(y) == length(x)
  copy!(y, x)
end

@cuda function do_forward!(n::ANode{:store!}, y::CudaArray, x)
  @assert length(y) == length(x)
  copy!(y, x; stream=STREAM[1].handle)
end

mutates(::ANode{:axpy!}) = true

function axpy!(y::ANode, x::ANode, alpha::Real=1)
  @assert size(x) == size(y)
  return store!(y, y + alpha*x)
end

function do_forward!(n::ANode{:axpy!}, y, x)
  @assert length(y) == length(x)
  BLAS.axpy!(length(y), convert(eltype(y), n.arg), x, 1, y, 1)
end
