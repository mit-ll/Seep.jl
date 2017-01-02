# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--gpu"
        help = "Use GPU (cuda)"
        action = :store_true
    "--profile"
        help = "Use profiler"
        action = :store_true
end
const args = parse_args(ARGS, s)
const gpu = args["gpu"]
const profile = args["profile"]

if gpu
    using CUDArt
    CUDArt.init(0)
    device(0)
    make_node(x...) = SeepNode(CudaArray(x...))
    get_data(x) = to_host(x)
else
    make_node(x...) = SeepNode(x...)
    get_data(x) = copy(x)
end

using Seep

const X = linspace(-1, 3, 100)
const Y = sin(X.*X)

const x = make_node(Float64, 1, 1)
const y = make_node(Float64, 1, 1)
ai = x

const v = Set{SeepNode}()

n1 = 100
const w1 = make_node(randn(n1, 1))
const b1 = make_node(zeros(n1, 1))
ai = tanh(w1*ai+b1)
link!(objective(1e-3*(w1.*w1)), v)

n2 = 100
const w2 = make_node(randn(n2, n1)/sqrt(n1))
const b2 = make_node(zeros(n2, 1))
ai = tanh(w2*ai+b2)
link!(objective(1e-3*(w2.*w2)), v)

const w3 = make_node(randn(1, n2)/sqrt(n2))
const b3 = make_node(zeros(1, 1))
ai = w3*ai+b3
link!(objective(1e-3*(w3.*w3)), v)

const e = ai-y
link!(objective(e.*e), v)

const a = ai

const E = Float64[]
;

fill!(w1.backward, 0)
fill!(b1.backward, 0)
fill!(w2.backward, 0)
fill!(b2.backward, 0)
fill!(w3.backward, 0)
fill!(b3.backward, 0)

update = momentum(1e-3, 0.8)

if gpu
  for n in v
    Seep.streamify!(n)
  end
end

open(f->graphviz(f, first(v)), "example.gv", "w")

n = 0

function step()
    global n

    for i in randperm(length(X))
        ni = n::Int + 1
        n = ni
        x[ni] = collect(X[i])
        y[ni] = Y[i:i]

        backward!(w1, ni)
        backward!(b1, ni)
        backward!(w2, ni)
        backward!(b2, ni)
        backward!(w3, ni)
        backward!(b3, ni)
    end

    push!(E, get_data(forward!(e, n))[1]^2)
    weight_update(update, w1, b1, w2, b2, w3, b3) 
end

step()

for i in 1:10
    @time step()
end

Profile.clear_malloc_data()

for i in 1:50
    step()
end

kill!(e)

if gpu
CUDArt.close(0)
end

println("finished cleaning up")
