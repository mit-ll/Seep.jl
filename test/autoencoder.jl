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
    "--minibatch"
        help = "Minibatch size"
        arg_type = Int
        default = 50
    "--epochs"
        help = "Epochs to train"
        arg_type = Int
        default = 100
    "--graphviz"
        help = "Dump the network structure to mnist.gv"
        action = :store_true
    "--64"
        help = "Use 64 bits"
        action = :store_true
end

const args = parse_args(ARGS, s)
const gpu = args["gpu"]
const profile = args["profile"]
const minibatch_size = args["minibatch"]
const epochs = args["epochs"]

if !gpu
  ENV["SEEP_NO_GPU"]=1
end

using MNIST, Seep, JLD

const T = args["64"] ? Float64 : Float32

const ftrain = map(T, traindata()[1]/255)

if gpu
    using CUDArt
    CUDArt.init(0)
    device(0)
    make_node(n, a::Array) = ANode(n, CudaArray(map(T, a)))
    make_node(n, x::Int...) = ANode(n, x...)
    get_data(x) = to_host(x)
    const pool = BuddyPool(CudaArray{T}(1<<24))
else
    make_node(n, a::Array) = ANode(n, map(T, a))
    make_node(n, x::Int...) = ANode(n, x...)
    get_data(x) = x
    const pool = BuddyPool(Array{T}(1<<24))
end

const n0 = 28*28
n = n0

const input = ANode("input", n, minibatch_size)
x = input

weights = ANode[]
parameters = ANode[]
for neurons = [50, 50, 2, 50, 50]
  A = make_node(randn(neurons, n)/sqrt(n))
  b = make_node(zeros(neurons, 1))
  x = A*x .+ b
  if neurons > 10
    x = tanh(x)
  end
  n = neurons

  push!(weights, A)
  push!(parameters, A)
  push!(parameters, b)
end

let neurons=28*28
  A = make_node(randn(neurons, n)/sqrt(n))
  b = make_node(zeros(neurons, 1))
  x = A*x .+ b
  n = neurons

  push!(weights, A)
  push!(parameters, A)
  push!(parameters, b)
end

const e = sum((x-input).^2)./2
const obj = e + 1e-3*reduce(+, [sum(A.^2) for A in weights])
const grad = gradients(obj)

upd = map(x->Seep.adam_fast(x, grad[x]), parameters)

const E = Float64[]

if args["graphviz"]
  #open(f->graphviz(f,  e, upd...), "mnist.gv", "w")
  open(f->graphviz(f,  e), "mnist.gv", "w")
end

Seep.devectorize(e, upd...)

const run,dict = instance(pool, e, upd...)
#const run,dict = instance(e, upd...)
const xa = dict[input]

#run() = global dict = evaluate(e, upd...)

Seep.print_stats(STDOUT, pool)

n = 0

function minibatch(ii)
    copy!(xa, ftrain[:,ii])
    run()
end

function epoch()
  perm = randperm(60000)
  ee = Float64[]
  for i in 1:minibatch_size:60000
    minibatch(perm[i+(1:minibatch_size)-1])
    push!(ee, sum(get_data(dict[e]))/minibatch_size)
  end
  push!(E, mean(ee))
end

epoch()

Profile.clear_malloc_data()

if profile
    Profile.init(delay=0.01)
    for i in 1:epochs
        @profile epoch()
    end

    data,lidict = Profile.retrieve()
    @save "mnist-profile.jld" data lidict
else
    for i in 1:epochs
        @time epoch()
        println(E[end])
    end
end

if gpu
  free(pool.pool)
  CUDArt.close(0)
end
