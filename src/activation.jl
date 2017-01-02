# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
for op in (:abs, :sign, :exp, :log, :tanh, :sqrt, :log1p, :expm1, :sigm)
  @eval function $(fsym(op))(x::ANode)
    ANode($(QuoteNode(op)), (x,), size(x))
  end
end

relu(x, leaky::Real=0) = max(leaky == 0 ? 0 : leaky*x, x)
softmax(x) = let z = exp(x); z./sum(z) end
softmax(x,dim) = let z = exp(x); z./sum(z, dim) end
softplus(x) = log(1+exp(x))

sigmoid_loss(x, y) = max(0,x) - x.*y + log(1 + exp(-abs(x))) # == -y.*log(sigm(x)) - (1-y).*log(1-sigm(x))

### sign

elementwise(::ANode{:sign}) = true

_sign{T}(n::Integer, a::T, b::T) = _sign(Csize_t(n), a, b)

function _sign{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = sign(b[i])
    end
end

@cuda_gsl _sign{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = (T(0) < b[i]) - (b[i] < T(0))"
@llvm llvm_val(n::ANode{:sign}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_copysign", t, t, t), [LLVM.ConstReal(TypeRef(t), 1), a], n.name)

do_forward!(n::ANode{:sign}, out, in) = _sign(length(out), out, in)

gradient_node(n::ANode{:sign}, wrt::ANode, b::ANode) = zeros(ANode, size(wrt))

### abs

elementwise(::ANode{:abs}) = true

_abs{T}(n::Integer, a::T, b::T) = _abs(Csize_t(n), a, b)

function _abs{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = abs(b[i])
    end
end

@cuda_gsl _abs{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = abs(b[i])"
@llvm llvm_val(n::ANode{:abs}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_abs", t, t), [a], n.name)

do_forward!(n::ANode{:abs}, out, in) = _abs(length(out), out, in)

gradient_node(n::ANode{:abs}, wrt::ANode, b::ANode) = b .* sign(wrt)

### sigm

sigm(x::AbstractArray) = map(sigm, x)
function sigm(x::Number)
    if x >= 0
        z = exp(-x)
        return 1 / (1 + z)
    else
        z = exp(x)
        return z / (1 + z)
    end
end

_sigm{T}(n::Integer, a::T, b::T) = _sigm(Csize_t(n), a, b)

function _sigm{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    @inbounds for i in 1:n
        a[i] = sigm(b[i])
    end
end

@cuda_kernel _sigm(n::Csize_t, a::&T, b::&T) [(T, Cfloat, Cdouble)] """
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
      if (b[i] >= 0) {
            T z = exp(-b[i]);
            a[i] = 1 / (1 + z);
        } else {
            T z = exp(b[i]);
            a[i] = z / (1 + z);
        }
    }
"""

do_forward!(n::ANode{:sigm}, out, in) = _sigm(length(out), out, in)

gradient_node(n::ANode{:sigm}, wrt::ANode, b::ANode) = b .* n .* (1 - n)


### exp

elementwise(::ANode{:exp}) = true
_exp{T}(n::Integer, a::T, b::T) = _exp(Csize_t(n), a, b)

function _exp{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = exp(b[i])
    end
end

@cuda_gsl _exp{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = exp(b[i])"
@llvm llvm_val(n::ANode{:exp}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_exp", t, t), [a], n.name)

do_forward!(n::ANode{:exp}, out, in) = _exp(length(out), out, in)

gradient_node(n::ANode{:exp}, wrt::ANode, b::ANode) = b .* n

### log

elementwise(::ANode{:log}) = true
# special case: turn log(1+x) into log1p(x)
Base.log(x::ANode{Symbol("c+")}) = arg(x) == 1 ? ANode(:log1p, (x.input[1],), size(x)) : ANode(:log, (x,), size(x))

_log{T}(n::Integer, a::T, b::T) = _log(Csize_t(n), a, b)

function _log{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = log(b[i])
    end
end

@cuda_gsl _log{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = log(b[i])"
@llvm llvm_val(n::ANode{:log}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_log", t, t), [a], n.name)

do_forward!(n::ANode{:log}, out, in) = _log(length(out), out, in)

gradient_node(n::ANode{:log}, wrt::ANode, b::ANode) = b ./ wrt

### log1p

elementwise(::ANode{:log1p}) = true
_log1p{T}(n::Integer, a::T, b::T) = _log1p(Csize_t(n), a, b)

function _log1p{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = log1p(b[i])
    end
end

@cuda_gsl _log1p{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = log1p(b[i])"
@llvm llvm_val(n::ANode{:log1p}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_log1p", t, t), [a], n.name)

do_forward!(n::ANode{:log1p}, out, in) = _log1p(length(out), out, in)

gradient_node(n::ANode{:log1p}, wrt::ANode, b::ANode) = b ./ (wrt+1)

### expm1

elementwise(::ANode{:expm1}) = true
_expm1{T}(n::Integer, a::T, b::T) = _expm1(Csize_t(n), a, b)

function _expm1{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = expm1(b[i])
    end
end

@cuda_gsl _expm1{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = expm1(b[i])"
@llvm llvm_val(n::ANode{:expm1}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_expm1", t, t), [a], n.name)

do_forward!(n::ANode{:expm1}, out, in) = _expm1(length(out), out, in)

gradient_node(n::ANode{:expm1}, wrt::ANode, b::ANode) = b .* (n+1)

### tanh

elementwise(::ANode{:tanh}) = true
_tanh{T}(n::Integer, a::T, b::T) = _tanh(Csize_t(n), a, b)

function _tanh{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = tanh(b[i])
    end
end

@cuda_gsl _tanh{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = tanh(b[i])"
@llvm llvm_val(n::ANode{:tanh}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_tanh", t, t), [a], n.name)

do_forward!(n::ANode{:tanh}, out, in) = _tanh(length(out), out, in)

gradient_node(n::ANode{:tanh}, wrt::ANode, b::ANode) = b.*(1 - n.^2)

### sqrt

elementwise(::ANode{:sqrt}) = true
_sqrt{T}(n::Integer, a::T, b::T) = _sqrt(Csize_t(n), a, b)

function _sqrt{T}(n::Csize_t, a::Array{T}, b::Array{T})
    @assert length(a) == length(b) == n
    for i in 1:n
        @inbounds a[i] = sqrt(b[i])
    end
end

@cuda_gsl _sqrt{T<:Real}(n::Csize_t, a::&T, b::&T) "a[i] = sqrt(b[i])"
@llvm llvm_val(n::ANode{:sqrt}, mod::ModuleRef, b::BuilderRef, t::Type, a::ValueRef) =
  LLVM.BuildCall(b, intrinsic(mod, "__nv_sqrt", t, t), [a], n.name)

do_forward!(n::ANode{:sqrt}, out, in) = _sqrt(length(out), out, in)

gradient_node(n::ANode{:sqrt}, wrt::ANode, b::ANode) = b./(2*n)
