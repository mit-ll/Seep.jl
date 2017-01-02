# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using Seep
using Base.Test
using CUDArt

CUDArt.init(0)

tocpu(x::Real) = x
togpu(x::Real) = x
tocpu(x::Array) = SeepNode(x)
togpu(x::Array) = SeepNode(CudaArray(x))
Base.vec(x::Real) = Float64[x]

function cmp(ex)
  @assert ex.head == :call
  op = ex.args[1]
  args = map(eval, ex.args[2:end])
  cpuargs = map(tocpu, args)

  jl = eval(:($op($args...)))
  cpu = eval(:($op($(cpuargs...))))
  gpu = eval(:($op($(map(togpu, args)...))))

  # compare cpu forward implementation to julia
  @assert norm(vec(forward!(cpu, 1))-vec(jl)) < 1e-6

  # compare gpu forward implementation to julia
  @assert norm(vec(to_host(forward!(gpu, 1)))-vec(jl)) < 1e-6

  kill!(cpu)
  kill!(gpu)
end

srand(1)

const N = 5

for op in [:+, :-]
  @show op
  cmp(:($op(randn(N,N), randn(N,N))))
end

for op in [:.+, :.-, :.*, :./]
  @show op
  cmp(:($op(randn(N,N), randn(N,N))))
  cmp(:($op(randn(1,N), randn(N,N))))
  cmp(:($op(randn(N,N), randn(1,N))))
  cmp(:($op(randn(N,1), randn(N,N))))
  cmp(:($op(randn(N,N), randn(N,1))))
  cmp(:($op(randn(N,1), randn(1,N))))
  cmp(:($op(randn(1,N), randn(N,1))))
end

for a in ("", "t", "c")
  for b in ("", "t", "c")
    op = a == b == "" ? :* : symbol("A", a, "_mul_B", b)
    if !isdefined(Base, op) continue end
    @show op
    cmp(:($op(randn(N,N), randn(N,N))))
  end
end

for op in [:exp, :log]
  @show op
  cmp(:($op(rand(N,N))))
  cmp(:($op(rand(1,N))))
  cmp(:($op(rand(N,1))))
end

for op in [:exp, :tanh, :sum, :softmax]
  @show op
  cmp(:($op(randn(N,N))))
  cmp(:($op(randn(1,N))))
  cmp(:($op(randn(N,1))))
end

for op in [:sum]
  @show op
  for i in 1:2
    cmp(:($op(randn(N,N), $i)))
    cmp(:($op(randn(1,N), $i)))
    cmp(:($op(randn(N,1), $i)))
  end
end
