# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
gradient_descent(x::ANode, g::ANode, alpha::Real=1e-3) = axpy!(x, g, -alpha)

function momentum(x::ANode, g::ANode, α::Real=1e-3, μ::Real=0.9)
    v = ANode(zeros(eltype(arg(x)), size(x))) # this is really v/α
    v1 = μ*v + g
    v2 = store!(v, v1)
    axpy!(x, v2, -α)
end

function adadelta(w::ANode, g::ANode, ϵ::Real=1e-6, ρ::Real=0.95)
  Eg² = ANode("E[g²]", zeros(eltype(arg(w)), size(w)))
  EΔ² = ANode("E[Δ²]", zeros(eltype(arg(w)), size(w)))

  u = store!(Eg², ρ*Eg² + (1-ρ)*(g.^2))
  Δ = g.*√((EΔ² + ϵ)./(u + ϵ))
  order(store!(EΔ², ρ*EΔ² + (1-ρ)*(Δ.^2)), axpy!(w, Δ, -1))
end

function adadelta_fast(w::ANode, g::ANode, ϵ::Real=1e-6, ρ::Real=0.95)
  Eg² = ANode("E[g²]", zeros(eltype(arg(w)), size(w)))
  EΔ² = ANode("E[Δ²]", zeros(eltype(arg(w)), size(w)))
  ANode(:adadelta, (w, g, Eg², EΔ²), size(w), (ϵ, ρ))
end

mutates(::ANode{:adadelta}) = true
function do_forward!(n::ANode{:adadelta}, w, g, Eg², EΔ²)
  ϵ = arg(n)[1]
  ρ = arg(n)[2]
  adadelta_update(Csize_t(length(w)), w, g, convert(eltype(w), ϵ), convert(eltype(w), ρ), Eg², EΔ²)
end

@cuda_kernel adadelta_update(n::Csize_t, v::&T, g::&T, e::T, r::T, Eg2::&T, ED2::&T) [(T, Cfloat, Cdouble)] """
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    Eg2[i] = r*Eg2[i] + (1-r)*(g[i]*g[i]);
    T D = g[i] * sqrt((ED2[i] + e)/(Eg2[i] + e));
    v[i] -= D;
    g[i] = 0;
    ED2[i] = r*ED2[i] + (1-r)*(D*D);
  }
"""

function adadelta_update{T}(n::Csize_t, v::Array{T}, g::Array{T}, ϵ::T, ρ::T, Eg²::Array{T}, EΔ²::Array{T})
  @assert n == length(v) == length(g) == length(g) == length(Eg²) == length(EΔ²)
  for i in 1:n
    Eg²[i] = ρ*Eg²[i] + (1-ρ)g[i].^2
    Δ = g[i] * √((EΔ²[i] + ϵ)/(Eg²[i] + ϵ))
    v[i] -= Δ
    EΔ²[i] = ρ*EΔ²[i] + (1-ρ)Δ.^2
  end
end


function adam(w::ANode, g::ANode, α::Real=1e-3, β1::Real=0.9, β2::Real=0.999, ϵ::Real=1e-8)
  t = ANode("t", ones(eltype(arg(w)), 1))
  m = ANode("m", zeros(eltype(arg(w)), size(w)))
  v = ANode("v", zeros(eltype(arg(w)), size(w)))

  αt = α*sqrt(1-β2.^t)./(1-β1.^t)

  sm = store!(m, β1*m + (1-β1)*g)
  sv = store!(v, β2*v + (1-β2)*(g.^2))

  order(store!(t,t+1), store!(w, w - αt.*(sm./(sqrt(sv)+ϵ))))
end

function adam_fast(w::ANode, g::ANode, α::Real=1e-3, β1::Real=0.9, β2::Real=0.999, ϵ::Real=1e-8)
  t = ANode("t", ones(eltype(arg(w)), 1))
  m = ANode("m", zeros(eltype(arg(w)), size(w)))
  v = ANode("v", zeros(eltype(arg(w)), size(w)))
  ANode(:adam, (w, g, t, m, v), size(w), (α, β1, β2, ϵ))
end

mutates(::ANode{:adam}) = true
function do_forward!(n::ANode{:adam}, w, g, t, m, v)
  α,β1,β2,ϵ = arg(n)
  adam_update(Csize_t(length(w)), w, g, convert(eltype(w), α), convert(eltype(w), β1), convert(eltype(w), β2), convert(eltype(w), ϵ), t, m, v)
end

@cuda_kernel adam_update(n::Csize_t, val::&T, g::&T, a::T, b1::T, b2::T, e::T, t::&T, m::&T, v::&T) [(T, Cfloat, Cdouble)] """
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    T gi = g[i];
    m[i] = b1*m[i] + (1-b1)*gi;
    v[i] = b2*v[i] + (1-b2)*(gi*gi);
    T at = a*sqrt(1-pow(b2,t[0]))/(1-pow(b1,t[0]));
    val[i] -= at*m[i]/(sqrt(v[i])+e);
    g[i] = 0;
  }
  if (i == 0) t[0] += 1;
"""

function adam_update{T}(n::Csize_t, val::Array{T}, g::Array{T}, α::T, β1::T, β2::T, ϵ::T, t::Array{T}, m::Array{T}, v::Array{T})
  @assert n == length(val) == length(g) == length(m) == length(v)
  αt = α*sqrt(1-β2^t[1])/(1-β1^t[1])
  t[1] += 1
  @inbounds for i in 1:n
    m[i] = β1*m[i] + (1-β1)*g[i]
    v[i] = β2*v[i] + (1-β2)*(g[i].^2)
    val[i] -= αt*m[i]/(sqrt(v[i])+ϵ)
  end
end

function rmsprop(w::ANode, g::ANode, α::Real=1e-4, ρ::Real=0.95, μ::Real=0.9, ϵ::Real=1e-6)
  Eg² = ANode("E[g²]", zeros(eltype(arg(w)), size(w)))
  u = store!(Eg², ρ*Eg² + (1-ρ)*(g.^2))
  w1 = store!(w, μ*w - α*g./√(u + ϵ))
  axpy!(w, w1, 1)
end

function rmsprop_fast(w::ANode, g::ANode, α::Real=1e-4, μ::Real=0.9, ρ::Real=0.95, ϵ::Real=1e-6)
  Eg² = ANode("E[g²]", zeros(eltype(arg(w)), size(w)))
  ANode(:rmsprop, (w, g, Eg²), size(w), (α, ρ, μ, ϵ))
end

mutates(::ANode{:rmsprop}) = true
function do_forward!(n::ANode{:rmsprop}, w, g, Eg²)
  α = arg(n)[1]
  ρ = arg(n)[2]
  μ = arg(n)[3]
  ϵ = arg(n)[4]
  rmsprop_update(Csize_t(length(w)), w, g, convert(eltype(w), α), convert(eltype(w), ρ), convert(eltype(w), μ), convert(eltype(w), ϵ), Eg²)
end

@cuda_kernel rmsprop_update(n::Csize_t, v::&T, g::&T, a::T, r::T, m::T, e::T, Eg2::&T) [(T, Cfloat, Cdouble)] """
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    Eg2[i] = r*Eg2[i] + (1-r)*(g[i]*g[i]);
    T D = m*v[i] - a * g[i] /sqrt(Eg2[i] + e);
    v[i] += D;
    g[i] = 0;
  }
"""

function rmsprop_update{T}(n::Csize_t, w::Array{T}, g::Array{T}, α::T, ρ::T, μ::T, ϵ::T, Eg²::Array{T})
  @assert n == length(w) == length(g) == length(Eg²)
  @inbounds for i in 1:n
    Eg²[i] = ρ*Eg²[i] + (1-ρ)*g[i]^2
    Δ = μ*w[i] - α * g[i]/sqrt(Eg²[i] + ϵ)
    w[i] = Δ
  end
end
