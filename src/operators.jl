# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
fsym(s::Symbol) = isdefined(Base, s) ? :(Base.$s) : s

for op in (:+, :-)
  @eval function $(fsym(op))(x::ANode, y::ANode)
    @assert size(x) == size(y)
    return ANode($(QuoteNode(Symbol(".",op))), (x, y), size(x))
  end
end

for op in (:.+, :.-, :.*, :./, :.^,
          :.==, :.!=, :.<, :.<=, :.>, :.>=)
  @eval function $(fsym(op))(x::ANode, y::ANode)
    if size(x) == size(y)
      return ANode($(QuoteNode(op)), (x, y), size(x))
    else
      return broadcast($op, x, y)
    end
  end
end

for op in [:+, :-, :*, :/]
  @eval begin
    Base.$(Symbol(".", op))(x::Real, y::ANode) = $(op)(x, y)
    Base.$(Symbol(".", op))(x::ANode, y::Real) = $(op)(x, y)
  end
end

@llvm begin
  llvm_val(n::ANode{:.+}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef,  a2::ValueRef) = LLVM.BuildFAdd(b, a1, a2, n.name)
  llvm_val(n::ANode{:.-}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef,  a2::ValueRef) = LLVM.BuildFSub(b, a1, a2, n.name)
  llvm_val(n::ANode{:(.*)}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef,  a2::ValueRef) = LLVM.BuildFMul(b, a1, a2, n.name)
  llvm_val(n::ANode{:(./)}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef,  a2::ValueRef) = LLVM.BuildFDiv(b, a1, a2, n.name)

  llvm_val(n::ANode{symbol("c+")}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef) = LLVM.BuildFAdd(b, a1, llvm_const(t, n.arg), n.name)
  llvm_val(n::ANode{symbol("c*")}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef) = LLVM.BuildFMul(b, a1, llvm_const(t, n.arg), n.name)
  llvm_val(n::ANode{symbol("c/")}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef) = LLVM.BuildFDiv(b, a1, llvm_const(t, n.arg), n.name)
  llvm_val(n::ANode{symbol("/c")}, ::ModuleRef, b::BuilderRef, t::Type, a1::ValueRef) = LLVM.BuildFDiv(b, llvm_const(t, n.arg), a1, n.name)

  llvm_val(n::ANode{symbol("^")}, m::ModuleRef, b::BuilderRef, t::Type{Float32}, a1::ValueRef, a2::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_powf", t, t, t), [a1, a2], n.name)
  llvm_val(n::ANode{symbol("^c")}, m::ModuleRef, b::BuilderRef, t::Type{Float32}, a1::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_powf", t, t, t), [a1, llvm_const(t, n.arg)], n.name)
  llvm_val(n::ANode{symbol("c^")}, m::ModuleRef, b::BuilderRef, t::Type{Float32}, a1::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_powf", t, t, t), [llvm_const(t, n.arg), a1], n.name)

  llvm_val(n::ANode{symbol("^")}, m::ModuleRef, b::BuilderRef, t::Type{Float64}, a1::ValueRef, a2::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_pow", t, t, t), [a1, a2], n.name)
  llvm_val(n::ANode{symbol("^c")}, m::ModuleRef, b::BuilderRef, t::Type{Float64}, a1::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_pow", t, t, t), [a1, llvm_const(t, n.arg)], n.name)
  llvm_val(n::ANode{symbol("c^")}, m::ModuleRef, b::BuilderRef, t::Type{Float64}, a1::ValueRef) =
    LLVM.BuildCall(b, intrinsic(m, "__nv_pow", t, t, t), [llvm_const(t, n.arg), a1], n.name)
end

function do_forward!(n::ANode{:.+}, out, in1, in2)
  copy!(out, in1)
  BLAS.axpy!(one(eltype(out)), in2, out)
end

@cuda function do_forward!(n::ANode{:.+}, out::CudaArray, in1, in2)
  copy!(out, in1; stream=STREAM[1].handle)
  BLAS.axpy!(one(eltype(out)), in2, out)
end

gradient_node(n::ANode{:.+}, wrt::ANode, b::ANode) = b

function do_forward!(n::ANode{:.-}, out, in1, in2)
  copy!(out, in1)
  BLAS.axpy!(-one(eltype(out)), in2, out)
end

@cuda function do_forward!(n::ANode{:.-}, out::CudaArray, in1, in2)
  copy!(out, in1; stream=STREAM[1].handle)
  BLAS.axpy!(-one(eltype(out)), in2, out)
end

gradient_node(n::ANode{:.-}, wrt::ANode, b::ANode) = wrt == n.input[1] ? b : -b

function eltmul0{T,N}(n::Csize_t, a::Array{T,N}, b::Array{T,N}, y::Array{T,N})
  @assert n == length(a) == length(b) == length(y)
  @simd for i in 1:n
    @inbounds y[i] = a[i]*b[i]
  end
  nothing
end
@cuda_gsl eltmul0{T<:Real}(n::Csize_t, a::&T, b::&T, y::&T) "y[i] = a[i]*b[i]"

function eltmul1{T,N}(n::Csize_t, a::Array{T,N}, b::Array{T,N}, alpha::T, y::Array{T,N})
  @assert n == length(a) == length(b) == length(y)
  @simd for i in 1:n
    @inbounds y[i] = alpha*y[i] + a[i]*b[i]
  end
  nothing
end
@cuda_gsl eltmul1{T<:Real}(n::Csize_t, a::&T, b::&T, alpha::T, y::&T) "y[i] = alpha*y[i] + a[i]*b[i]"

function eltmul{T}(n::ANode, a::T, b::T, alpha::Real, y::T)
  if alpha == 0
    eltmul0(Csize_t(length(a)), a, b, y)
  else
    eltmul1(Csize_t(length(a)), a, b, convert(eltype(T), alpha), y)
  end
end


function do_forward!(n::ANode{:.*}, out, in1, in2)
  eltmul(n, in1, in2, 0, out)
end

gradient_node(n::ANode{:.*}, wrt::ANode, b::ANode) = b .* (wrt == n.input[1] ? n.input[2] : n.input[1])

function Base.dot(x::ANode, y::ANode)
  @assert size(x) == size(y)
  return ANode(:dot, (x, y), (1,))
end

function do_forward!(n::ANode{:dot}, out, in1, in2)
  fill!(out, BLAS.dot(length(in1), in1, 1, in2, 1))
end

gradient_node(n::ANode{:dot}, wrt::ANode, b::ANode) = b.*(wrt == n.input[1] ? n.input[2] : n.input[1])

function fprop_eltdiv{T}(n::Csize_t, y::Array{T}, a::Array{T}, b::Array{T})
    @assert length(y) == length(a) == length(b) == n
    for i in 1:n
      @inbounds y[i] = a[i]/b[i]
    end
end

@cuda_gsl fprop_eltdiv{T<:Real}(n::Csize_t, y::&T, fa::&T, fb::&T) "y[i] = fa[i]/fb[i]"

function do_forward!(n::ANode{:./}, out, in1, in2)
    fprop_eltdiv(Csize_t(length(out)), out, in1, in2)
end

gradient_node(n::ANode{:./}, wrt::ANode, b::ANode) = (wrt == n.input[1] ? b./n.input[2] : -b.*n.input[1]./(n.input[2].^2))

import Base.+
+(y::Real, x::ANode) = +(x,y)
+(x::ANode, y::Real) = ANode(Symbol("c+"), (x,), size(x), y)

# Special case transform exp(x)-1 into expm1(x)
+(x::ANode{:exp}, y::Real) = y == -1 ? ANode(:expm1, (x.input[1],), size(x)) : ANode(Symbol("c+"), (x,), size(x), y)

function do_forward!(n::ANode{Symbol("c+")}, out, in)
  fill!(out, n.arg)
  BLAS.axpy!(one(eltype(out)), in, out)
end

gradient_node(n::ANode{Symbol("c+")}, wrt::ANode, b::ANode) = b

import Base.-
-(x::ANode) = -1*x
-(x::Real, y::ANode) = +(x, -y)
-(x::ANode, y::Real) = +(x, -y)

import Base.*
*(y::Real, x::ANode) = *(x,y)
*(x::ANode, y::Real) = ANode(Symbol("c*"), (x,), size(x), y)

function do_forward!(n::ANode{Symbol("c*")}, out, in)
 copy!(out, in)
 BLAS.scal!(length(out), convert(eltype(out), n.arg), out, 1)
end

@cuda function do_forward!(n::ANode{Symbol("c*")}, out, in)
 copy!(out, in; stream=STREAM[1].handle)
 BLAS.scal!(length(out), convert(eltype(out), n.arg), out, 1)
end

gradient_node(n::ANode{Symbol("c*")}, wrt::ANode, b::ANode) = n.arg .* b

import Base./
/(x::ANode, y::Real) = x*(1/y)
/(x::Real, y::ANode) = ANode(Symbol("c/"), (y,), size(y), x)

function fprop_cdiv{T,N}(n::Csize_t, y::Array{T,N}, a::T, b::Array{T,N})
    @assert length(y) == length(b) == n
    for i in 1:n
      y[i] = a/b[i]
    end
end

@cuda_gsl fprop_cdiv{T<:Real}(n::Csize_t, y::&T, fa::T, fb::&T) "y[i] = fa/fb[i]"

function do_forward!(n::ANode{Symbol("c/")}, out, in)
  fprop_cdiv(Csize_t(length(out)), out, convert(eltype(out), n.arg), in)
end

gradient_node(n::ANode{Symbol("c/")}, wrt::ANode, b::ANode) = b .* -n.arg[1] ./ (n.input[1] .^ 2)

# pow
function fprop_pow{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::Array{T,N})
    @assert length(y) == length(a) == length(b) == n
    for i in 1:n
      y[i] = a[i]^b[i]
    end
end

@cuda_gsl fprop_pow{T<:Real}(n::Csize_t, y::&T, fa::&T, fb::&T) "y[i] = pow(fa[i], fb[i])"

function fprop_cpow{T,N}(n::Csize_t, y::Array{T,N}, a::T, b::Array{T,N})
    @assert length(y) == length(b) == n
    for i in 1:n
      y[i] = a^b[i]
    end
end

@cuda_gsl fprop_cpow{T<:Real}(n::Csize_t, y::&T, fa::T, fb::&T) "y[i] = pow(fa, fb[i])"

function fprop_powc{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::T)
    @assert length(y) == length(a) == n
    for i in 1:n
      y[i] = a[i]^b
    end
end

@cuda_gsl fprop_powc{T<:Real}(n::Csize_t, y::&T, fa::&T, fb::T) "y[i] = pow(fa[i], fb)"

import Base.(.^)
.^(x::ANode, y::Real) = ANode(Symbol("^c"), (x,), size(x), y)
.^(y::Real, x::ANode) = ANode(Symbol("c^"), (x,), size(x), y)

do_forward!(n::ANode{:.^}, out, in1, in2) = fprop_pow(Csize_t(length(out)), out, in1, in2)
do_forward!(n::ANode{Symbol("c^")}, out, in) = fprop_cpow(Csize_t(length(out)), out, convert(eltype(out), arg(n)), in)
do_forward!(n::ANode{Symbol("^c")}, out, in) = fprop_powc(Csize_t(length(out)), out, in, convert(eltype(out), arg(n)))

gradient_node(n::ANode{:.^}, wrt::ANode, b::ANode) = b.*(wrt == n.input[1] ? n.input[2].*(n.input[1].^(n.input[2]-1)) : n.*log(n.input[1]))
gradient_node(n::ANode{Symbol("c^")}, wrt::ANode, b::ANode) = b.*(n.*log(n.input[1]))
gradient_node(n::ANode{Symbol("^c")}, wrt::ANode, b::ANode) = b.*(arg(n).*(n.input[1].^(arg(n)-1)))

## .==

for (op,fname) in [
  (:(==), :_eq), (:(!=), :_ne),
  (:(<), :_lt), (:(<=), :_le),
  (:(>), :_gt), (:(>=), :_ge),
  ] @eval begin
      Base.$op(x::ANode, y::Real) = ANode(Symbol("c",$op), (x,), size(x), y)
      Base.$op(y::Real, x::ANode) = ANode(Symbol("c",$op), (x,), size(x), y)
      Base.$(Symbol(".",op))(x::ANode, y::Real) = ANode(Symbol("c",$op), (x,), size(x), y)
      Base.$(Symbol(".",op))(y::Real, x::ANode) = ANode(Symbol("c",$op), (x,), size(x), y)

      do_forward!(n::ANode{$(QuoteNode(Symbol(".",op)))}, out, in1, in2) = ($fname)(Csize_t(length(out)), out, in1, in2)
      do_forward!(n::ANode{$(QuoteNode(Symbol("c",op)))}, out, in1) = ($fname)(Csize_t(length(out)), out, in1, eltype(in1)(n.arg))

    function $fname{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::Array{T,N})
        @assert length(y) == length(a) == length(b) == n
        for i in 1:n
          y[i] = $op(a[i],b[i]) ? one(T) : zero(T)
        end
    end

    function $fname{T,N}(n::Csize_t, y::Array{T,N}, a::Array{T,N}, b::T)
        @assert length(y) == length(a) == n
        for i in 1:n
          y[i] = $op(a[i],b) ? one(T) : zero(T)
        end
    end

    @cuda_gsl $fname{T<:Real}(n::Csize_t, y::&T, fa::&T, fb::&T) "y[i] = fa[i] $($op) fb[i] ? 1 : 0"
    @cuda_gsl $fname{T<:Real}(n::Csize_t, y::&T, fa::&T, fb::T) "y[i] = fa[i] $($op) fb ? 1 : 0"

    gradient_node(n::ANode{$(QuoteNode(Symbol(".",op)))}, wrt::ANode, b::ANode) = zeros(ANode, size(wrt))
    gradient_node(n::ANode{$(QuoteNode(Symbol("c",op)))}, wrt::ANode, b::ANode) = zeros(ANode, size(wrt))
  end
end
