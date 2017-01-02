# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
VALID_ID = Union{Integer, AbstractString, Symbol}
node_name(n::AbstractString, k) = "$(n)_$(k)"

abstract Flow
get_params{T<:Flow}(creek::T) = error("not implemented ($T)")
get_params{T<:Flow}(creeks::AbstractArray{T}) = vcat([get_params(c) for c in creeks]...)
reg_params{T<:Flow}(creek::T) = error("not implemented ($T)")
reg_params{T<:Flow}(creeks::AbstractArray{T}) = vcat([reg_params(c) for c in creeks]...)
#@compat (creek::T){T<:Flow}(args...) = error("not implemented")
Base.size(creek::Flow, args...) = error("not implemented")
Base.eltype(creek::Flow) = error("not implemented")

in_size{T<:Flow}(creek::T) = error("not implemented ($T)")
out_size{T<:Flow}(creek::T) = error("not implemented ($T)")

function scanl{T}(f, ::Type{T}, a0::T, xa)
  y = Array{T}(size(xa))
  a = a0
  for i in 1:length(xa)
    y[i] = a = f(a, xa[i])
  end
  return y
end

function scanl!(f, a0, xa)
  a = a0
  for i in 1:length(xa)
    xa[i] = a = f(a, xa[i])
  end
  return xa
end

# An array of flows behaves as composition
@compat (layers::Array{T}){T<:Flow}(x; kw...) = foldl((x, l)->l(x), x, layers)

# Any unary function (e.g. tanh, sigm) may be used as a flow
immutable Activation{F} <: Flow f::F end
Base.convert{F<:Function}(::Type{Flow}, f::F) = Seep.Activation(f)
Base.convert{F<:Function}(::Type{Activation}, f::F) = Seep.Activation{F}(f)
@compat (a::Activation)(x) = a.f(x)
@compat (a::Activation)(xs::AbstractArray) = map(a.f, xs)

get_params(::Activation) = ANode[]
reg_params(::Activation) = ANode[]

# Uniform Xavier Initialization
function xavier{T}(::Type{T}, shape...; scale=shape[end])
    @assert scale>0
    val = map(T, rand(shape...) / sqrt(scale))
    !haskey(ENV, "SEEP_NO_GPU") ?  CudaArray(val) : val
end
xavier{T}(::Type{T}, nout) = xavier(T, nout, 1)

# Gaussian (Normal) Xavier Initialization
function xaviern{T}(::Type{T}, shape...; scale=shape[end])
    @assert scale>0
    val = map(T, randn(shape...) / sqrt(scale))
    !haskey(ENV, "SEEP_NO_GPU") ?  CudaArray(val) : val
end
xaviern{T}(::Type{T}, nout) = xaviern(T, nout, 1)

function zero_init{T}(::Type{T}, s...)
    val = map(T, zeros(s...))
    !haskey(ENV, "SEEP_NO_GPU") ?  CudaArray(val) : val
end

type Linear <: Flow
    name::VALID_ID
    W::ANode
    b::ANode
end
function Linear{T}(::Type{T}, k, input_size::Integer, output_size::Integer;
    W=xaviern(T, output_size, input_size), b=zero_init(T, output_size))
    Linear(k, ANode(node_name("W", k), W), ANode(node_name("b", k), b))
end
Base.size(l::Linear, args...) = size(l.W, args...)
Base.eltype(l::Linear) = eltype(l.W)
get_params(l::Linear) = ANode[l.W, l.b]
reg_params(l::Linear) = ANode[l.W]
in_size(l::Linear) = size(l.W, 2)
out_size(l::Linear) = size(l.W, 1)
@compat (l::Linear)(x) = l.W * x .+ l.b
@compat (l::Linear)(xs::AbstractArray) = map(l, xs)

type Dropout <: Flow
    a::Real
end
get_params(::Dropout) = ANode[]
reg_params(::Dropout) = ANode[]
@compat (d::Dropout)(x) = (rand!(ANode(size(x))) .> d.a).*x./(1-d.a)
@compat (d::Dropout)(x::AbstractArray) = map(d, x)

type RandomNoise <: Flow
    a::Real
end
get_params(::RandomNoise) = ANode[]
reg_params(::RandomNoise) = ANode[]
@compat (d::RandomNoise)(x) = x.+(randn!(ANode(size(x))))
@compat (d::RandomNoise)(x::AbstractArray) = map(d, x)

type Conv2D <: Flow
    name::VALID_ID
    W::ANode
    b::ANode
    filter_shape
    stride
end
function Conv2D{T}(::Type{T}, k, filter_shape::Tuple{Vararg{Int}}; stride=(),
    W=xaviern(T, filter_shape...; scale=prod(filter_shape[2:end])),
    b=zero_init(T, filter_shape[1]))

    Conv2D(k, ANode(node_name("W", k), W), ANode(node_name("b", k), b), filter_shape, stride)
end
Base.size(l::Conv2D, args...) = size(l.W, args...)
Base.eltype(l::Conv2D) = eltype(l.W)
get_params(l::Conv2D) = ANode[l.W, l.b]
reg_params(l::Conv2D) = ANode[]
@compat (l::Conv2D)(x) = conv2(l.W, x; stride=l.stride) .+ l.b
@compat (l::Conv2D)(xs::AbstractArray) = map(x->l(x), xs)

type Pool <: Flow
    filter_shape
    stride
end
function Pool(filter_shape::Tuple{Vararg{Int}}; stride=filter_shape)
    Pool(filter_shape, stride)
end
get_params(l::Pool) = ANode[]
reg_params(l::Pool) = ANode[]
@compat (l::Pool)(x) = pool(x, l.filter_shape; stride=l.stride)
@compat (l::Pool)(xs::AbstractArray) = map(x->l(x), xs)

type RNN <: Flow
    name::VALID_ID
    Wx::ANode
    Wh::ANode
    b::ANode
end
function RNN{T}(::Type{T}, k, input_size::Integer, output_size::Integer;
    Wx=xaviern(T, output_size, input_size),
    Wh=xaviern(T, output_size, input_size),
    b=zero_init(T, output_size))

    RNN(k, ANode(node_name("Wx", k), Wx), ANode(node_name("Wh", k), Wh), ANode(node_name("b", k), b))
end
Base.size(l::RNN, args...) = size(l.Wx, args...)
Base.eltype(l::RNN) = eltype(l.Wx)
get_params(l::RNN) = ANode[l.Wx, l.Wh, l.b]
reg_params(l::RNN) = ANode[l.Wx, l.Wh]
in_size(l::RNN) = size(l.Wx, 2)
out_size(l::RNN) = size(l.Wx, 1)
@compat (l::RNN)(x, h) = l.Wx*x .+ l.Wh*h .+ l.b
@compat (l::RNN)(xs::AbstractArray, h0=h0_default(l, xs)) = scanl(l, ANode, h0, xs)
h0_default(l::Flow, xs::AbstractArray) = ANode(:zeros, (), (out_size(l), size(xs[1], 2)))


type LSTM <: Flow
    name::VALID_ID
    # Weights
    Wxa::ANode
    Wha::ANode
    ba::ANode
    Wxi::ANode
    Whi::ANode
    bi::ANode
    Wxf::ANode
    Whf::ANode
    bf::ANode
    Wxo::ANode
    Who::ANode
    bo::ANode
end
function LSTM{T}(::Type{T}, k, input_size, output_size; 
    Wx=xaviern(T, 4*output_size, input_size), Wh=xaviern(T, 4*output_size, output_size), b=zero_init(T, 4*output_size))

    ind = 1:output_size
    Wxa = ANode(node_name("Wxa", k), Wx[ind, :])
    Wha = ANode(node_name("Wha", k), Wh[ind, :])
    ba = ANode(node_name("ba", k), b[ind])

    ind = ind[end] + (1:output_size)
    Wxi = ANode(node_name("Wxi", k), Wx[ind, :])
    Whi = ANode(node_name("Whi", k), Wh[ind, :])
    bi = ANode(node_name("bi", k), b[ind])
    
    ind = ind[end] + (1:output_size)
    Wxf = ANode(node_name("Wxf", k), Wx[ind, :])
    Whf = ANode(node_name("Whf", k), Wh[ind, :])
    bf = ANode(node_name("Wbf", k), b[ind])
    
    ind = ind[end] + (1:output_size)
    Wxo = ANode(node_name("Wxo", k), Wx[ind, :])
    Who = ANode(node_name("Who", k), Wh[ind, :])
    bo = ANode(node_name("bo", k), b[ind])
    
    LSTM(k, Wxa, Wha, ba, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo)
end 
Base.size(l::LSTM, args...) = size(l.Wxa, args...)
Base.eltype(l::LSTM) = eltype(l.Wxa)
get_params(l::LSTM) = begin
    params = ANode[]
    push!(params, l.Wxa, l.Wha, l.ba)
    push!(params, l.Wxi, l.Whi, l.bi)
    push!(params, l.Wxf, l.Whf, l.bf)
    push!(params, l.Wxo, l.Who, l.bo)
    params
end
reg_params(l::LSTM) = begin
    params = ANode[]
    push!(params, l.Wxa, l.Wha)
    push!(params, l.Wxi, l.Whi)
    push!(params, l.Wxf, l.Whf)
    push!(params, l.Wxo, l.Who)
    params
end
in_size(l::LSTM) = size(l.Wxa, 2)
out_size(l::LSTM) = size(l.Wxa, 1)

function apply_lstm(l::LSTM, x, c, h)
    a = tanh(l.Wxa * x .+ l.Wha * h .+ l.ba)
    i = sigm(l.Wxi * x .+ l.Whi * h .+ l.bi)
    f = sigm(l.Wxf * x .+ l.Whf * h .+ l.bf)
    o = sigm(l.Wxo * x .+ l.Who * h .+ l.bo)
    c = a .* i + f.* c
    h = o .* tanh(c)
    c, h
end

apply_lstm(l::LSTM, xs::AbstractArray, ch=(h0_default(l, xs), h0_default(l, xs))) =
  scanl((ch,x)->apply_lstm(l, x, ch[1], ch[2]), Tuple{ANode,ANode}, ch, xs)

@compat (l::LSTM)(x) = error("LSTM should be applied to an array of input nodes")
@compat (l::LSTM)(xs::AbstractArray) = map(x->x[2], apply_lstm(l, xs))
@compat (l::LSTM)(xs::AbstractArray, ch) = map(x->x[2], apply_lstm(l, xs, ch))

type GRU <: Flow
    name::VALID_ID
    Wr::ANode
    Ur::ANode
    br::ANode
    Wz::ANode
    Uz::ANode
    bz::ANode
    W::ANode
    U::ANode
    b::ANode
end
function GRU{T}(::Type{T}, k, n, args...)
    Wr = ANode(node_name("Wr", k), xaviern(T, n, n, 2*n))
    Ur = ANode(node_name("Ur", k), xaviern(T, n, n, 2*n))
    br = ANode(node_name("br", k), zero_init(T, n))
    Wz = ANode(node_name("Wz", k), xaviern(T, n, n, 2*n))
    Uz = ANode(node_name("Uz", k), xaviern(T, n, n, 2*n))
    bz = ANode(node_name("bz", k), zero_init(T, n))
    W = ANode(node_name("W", k), xaviern(T, n, n, 2*n))
    U = ANode(node_name("U", k), xaviern(T, n, n, 2*n))
    b = ANode(node_name("b", k), zero_init(T, n))

    GRU(k, Wr, Ur, br, Wz, Uz, bz, W, U, b)
end
Base.size(l::GRU, args...) = size(l.Wxr, args...)
Base.eltype(l::GRU) = eltype(l.Wr)
get_params(gru::GRU) = begin
    params = ANode[]
    push!(params, gru.Wr, gru.Ur, gru.br)
    push!(params, gru.Wz, gru.Uz, gru.bz)
    push!(params, gru.W, gru.U, gru.b)
    params
end
reg_params(gru::GRU) = begin
    params = ANode[]
    push!(params, gru.Wr, gru.Ur)
    push!(params, gru.Wz, gru.Uz)
    push!(params, gru.W, gru.U)
    params
end
in_size(l::GRU) = out_size(l)
out_size(l::GRU) = size(l.Wr, 1)
@compat (g::GRU)(x, h) = begin
    r = sigm(g.Wr * x .+ g.Ur * h .+ g.br)
    z = sigm(g.Wz * x .+ g.Uz * h .+ g.bz)
    hbar = tanh(g.W * x .+ g.U * (r .* h) .+ g.b)
    (1-z) .* g + z .* hbar
end
@compat (g::GRU)(xs::AbstractArray, h=h0_default(g, xs)) = scanl(g, ANode, h, xs)
