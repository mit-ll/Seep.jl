# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using Base.Test

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--gpu"
        help = "Use GPU (cuda)"
        action = :store_true
end
const args = parse_args(ARGS, s)
const gpu = args["gpu"]

global compare_cpu_gpu
const T = Array{Float64}

if gpu
    using CUDArt
    CUDArt.init(0)
    device(0)
    Base.rand!(a::CudaArray) = copy!(a, rand(size(a)))
    Base.randn!(a::CudaArray) = copy!(a, randn(size(a)))
    function compare_cpu_gpu(cpu, gpu, node, input; tol=1e-5)
      set_values!(gpu, input)
      gpu()

      #if vecnorm(cpu[node] - to_host(gpu[node])) > tol
      #  @show cpu[node]
      #  @show to_host(gpu[node])
      #  @show cpu[node] - to_host(gpu[node])
      #  @show vecnorm(cpu[node] - to_host(gpu[node]))
      #  @show Seep.sym(node)
      #  Seep.graphviz(STDOUT, node)
      #  for a in gpu.pool.arrays
      #    @show to_host(a)
      #  end
      #end

      @test vecnorm(cpu[node] - to_host(gpu[node])) < tol
    end
else
    ENV["SEEP_NO_GPU"]="1"
    compare_cpu_gpu(x...; kw...) = nothing
end

using Seep
import Seep: sym, arg

Base.randn!(a::Array{Float32}) = copy!(a, randn(size(a)))
mrandn!(as...) = for a in as randn!(a) end

set_values!(inst::Seep.Instance, vals) = for (n,v) in vals copy!(inst[n], v) end

function runtest(y::ANode, x::Vector; fill!::Function=mrandn!, jop=nothing, v_tol=1e-5, d_tol=1e-5, delta=1e-3)
  srand(2)

  grad = gradients(y)
  cpu = instance(T,[y; x; map(a->grad[a], x)])
  igpu = gpu ? instance(CudaArray{eltype(T)}, [y; x; map(a->grad[a], x)]) : nothing

  for i in 1:25
    fill!(map(a->cpu[a], x)...)
    x0 = [(a,copy(cpu[a])) for a in x]
    cpu()

    y0 = copy(cpu[y])

    if jop != nothing
      yref = @eval $(jop)($(map(a->cpu[a], x))...)
      @test !(vecnorm(y0 - yref) > v_tol)
    end

    compare_cpu_gpu(cpu, igpu, y, x0)

    for a in x
      set_values!(cpu, x0)

      a0 = copy(cpu[a])
      gr = copy(cpu[grad[a]])

      cpu()
      compare_cpu_gpu(cpu, igpu, grad[a], x0)

      for j in 1:length(a0)
        set_values!(cpu, x0)

        dx = zeros(eltype(T), size(a0))
        dx[j] += delta*randn()

        copy!(cpu[a], a0+dx)
        cpu()

        if (sum(y0) + sum(gr.*dx) - sum(cpu[y]) > d_tol)
          @show a0
          @show y0
          @show gr
          @show dx
          @show gr.*dx
          @show cpu[y]
          @show sum(y0) + sum(gr.*dx) - sum(cpu[y])
        end
        @test !(sum(y0) + sum(gr.*dx) - sum(cpu[y]) > d_tol)
      end
    end
  end

  if gpu
    close(igpu._pool)
  end
end

println("5x5")
for op in [:+, :.+, :-, :.-, :.*, :*, :At_mul_B, :A_mul_Bt, :At_mul_Bt]
    @show op
    a = ANode(5, 5)
    b = ANode(5, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b], jop=op)
end

for op in [:./, :.^]
    @show op
    a = ANode(5, 5)
    b = ANode(5, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b], fill! =(a,b)->(rand!(a); copy!(b, rand(size(b))+1)), jop=op, delta=1e-5)
end

for op in [:dot]
    @show op
    a = ANode(5, 5)
    b = ANode(5, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b])
end

println("5x5 ⊙ 5x1")
for op in [:.+, :.-, :.*, :*, :At_mul_B]
    @show op
    a = ANode(5, 5)
    b = ANode(5, 1)
    y = @eval $op($a, $b)
    runtest(y, [a, b], jop=op)
end

println("5x5 ⊙ 1x5")
for op in [:.+, :.-, :.*, :At_mul_Bt, :A_mul_Bt]
    @show op
    a = ANode(5, 5)
    b = ANode(1, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b], jop=op)
end

println("5x1 ⊙ 5x5")
for op in [:.+, :.-, :.*, :At_mul_B, At_mul_Bt]
    @show op
    a = ANode(5, 1)
    b = ANode(5, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b], jop=op)
end

println("1x5 ⊙ 5x5")
for op in [:.+, :.-, :.*, :*, :A_mul_Bt]
    @show op
    a = ANode(1, 5)
    b = ANode(5, 5)
    y = @eval $op($a, $b)
    runtest(y, [a, b], jop=op)
end

println("vectors")
for op in [:exp, :log, :tanh, :sqrt, :sum, :sigm, :log1p, :expm1, :abs, :sign]
    @show op
    a = ANode(1, 5)
    y = @eval $op($a)
    runtest(y, [a], fill! = (a)->(rand!(a); a[:] += 0.5), jop=op, d_tol=1e-3, v_tol=1e-3)
end

for op in [:softmax]
    @show op
    a = ANode(1, 5)
    y = @eval $op($a)
    runtest(y, [a])
end

println("x,::Real")
for op in [:+, :-, :.+, :.-, :*, :.*]
    @show op

    a = ANode(5, 5)
    b = randn()
    y = @eval $op($a, $b)
    runtest(y, [a])

    a = ANode(5, 5)
    b = randn()
    y2 = @eval $op($b, $a)
    runtest(y2, [a])
end

println("getindex")
for index in (collect(1:5), reshape(25:-1:1, 5, 5))
    a = ANode(5, 5)
    y = a[index]
    runtest(y, [a])
end

println("reduce")
for f in [:sum, :minimum, :maximum]
    @show f

    a = ANode(5, 5)
    y = @eval $f($a)
    runtest(y, [a], delta=1e-5, d_tol=1e-6)

    y = @eval $f($a, 1)
    runtest(y, [a], delta=1e-5, d_tol=1e-6)

    y = @eval $f($a, 2)
    runtest(y, [a], delta=1e-5, d_tol=1e-6)
end

println("extrema")
for f in [:min!, :max!]
    @show f
    a = ANode(5, 5)

    y = @eval Seep.$f($a, 0)
    runtest(y, [a], delta=1e-5, d_tol=1e-5)
end

for f in [:min, :max]
    @show f

    a = ANode(5, 5)
    b = ANode(5, 5)

    y = @eval Seep.$f($a, 0)
    runtest(y, [a], delta=1e-5, d_tol=1e-6)

    y = @eval Seep.$f($a, $b)
    runtest(y, [a, b], delta=1e-5, d_tol=1e-6)
end

let
    a = ANode(1, 10, 10, 9)
    b = ANode(2, 1, 3, 3)

    @show :conv2
    y = conv2(b, a)
    runtest(y, [a,b], delta=1e-5, d_tol=1e-6)

    y = conv2(b, a, padding=(2,2,2))
    runtest(y, [a,b], delta=1e-5, d_tol=1e-6)

    @show :pool
    y = Seep.pool(a, (1,2,2))
    runtest(y, [a], delta=1e-5, d_tol=1e-6)

    y = Seep.pool(a, (1,2,2), padding=(2,2,2))
    runtest(y, [a], delta=1e-5, d_tol=1e-1)
end

if gpu
  CUDArt.close(0)
end
