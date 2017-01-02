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
        default = 100
    "--epochs"
        help = "Epochs to train"
        arg_type = Int
        default = 10
    "--devectorize"
        help = "Devectorize the graph before evaluation"
        action = :store_true
    "--test"
        help = "Evaluate test set"
        action = :store_true
    "--seed"
        help = "Random seed"
        arg_type = Int
    "--graphviz"
        help = "Dump the network structure to mnist.gv"
        action = :store_true
    "--load"
        help = "Load a snapshot before training"
        #arg_type = Int
    "--store"
        help = "Store a snapshot after training"
        #arg_type = Int
    "--64"
        help = "Use 64 bits"
        action = :store_true
    "--solver"
        help = "Select solver (sgd, momentum, adadelta, adam)"
        default = "momentum"
end

const args = parse_args(ARGS, s)
const gpu = args["gpu"]
const profile = args["profile"]
const minibatch_size = args["minibatch"]
const epochs = args["epochs"]
const solver = args["solver"]

if !gpu
  ENV["SEEP_NO_GPU"]=1
end

using MNIST, Seep, JLD

const T = args["64"] ? Float64 : Float32

const ftrain = map(T, traindata()[1]/255)
const ltrain = map(T, traindata()[2])
const ytrain = zeros(T, 10, length(ltrain))
for i in 1:size(ytrain, 2)
  ytrain[1+round(Int, ltrain[i]),i] = 1
end

if gpu
    using CUDArt
    CUDArt.init(0)
    device(0)
    make_node(n, a::Array) = ANode(n, CudaArray(map(T, a)))
    make_node(n, x::Int...) = ANode(n, x...)
    get_data(x) = to_host(x)
    const pool = BuddyPool(CudaArray{T}(1<<26))
else
    make_node(n, a::Array) = ANode(n, map(T, a))
    make_node(n, x::Int...) = ANode(n, x...)
    get_data(x) = x
    const pool = BuddyPool(Array{T}(1<<26))
end

if args["seed"] != nothing
  srand(args["seed"])
end

const x = make_node("x", 1, 28, 28, minibatch_size)
a = x

const w1 = make_node("w1", randn(8, 1, 5, 5)/sqrt(25))
const b1 = make_node("b1", zeros(8))
a = Seep.max!(0, conv2(w1, a) .+ b1)
a = Seep.pool(a, (1,2,2))

const w2 = make_node("w2", randn(16, 8, 5, 5)/sqrt(25*8))
const b2 = make_node("b2", zeros(16))
a = Seep.max!(0, conv2(w2, a) .+ b2)
a = Seep.pool(a, (1,2,2))

a = reshape(a, (prod(size(a)[1:end-1]), size(a)[end]))

const w3 = make_node("w3", randn(10, size(a, 1))/sqrt(size(a, 1)))
const b3 = make_node("b3", zeros(10))
a = softmax(w3*a .+ b3, 1)

const y = make_node("y", 10, minibatch_size)
const e = -log(a).*y

const obj= sum(e)

const grad = gradients(obj)

if solver == "sgd"
  const sol = gradient_descent
elseif solver == "momentum"
  const sol = momentum
elseif solver == "adadelta"
  const sol = Seep.adadelta
elseif solver == "adam"
  const sol = adam
else
  error("Unknown solver: $solver")
end

upd = map(x->sol(x, grad[x]), [w1, b1, w2, b2, w3, b3])

const E = Float64[]

if args["graphviz"]
  open(f->graphviz(f,  e, upd...), "mnist.gv", "w")
end

if args["devectorize"]
  Seep.devectorize(e, upd...)
end

const run,dict = instance(pool, e, a, upd...)

Seep.print_stats(STDOUT, pool)

if args["load"] !== nothing
  load_snapshot(args["load"], e)
end

n = 0

const xa,ya = dict[x],dict[y]

function minibatch(ii)
    if gpu
       copy!(xa, ftrain[:,ii])
       copy!(ya, ytrain[:,ii])
    else
        for i in 1:length(ii)
          for j in 1:(28*28)
            xa[(i-1)*28*28+j] = ftrain[j,ii[i]]
          end

          for j in 1:10
            ya[j,i] = ytrain[j,ii[i]]
          end
        end
    end

    run()

    push!(E, sum(get_data(dict[e]))/length(ii))

end

function epoch()
  perm = randperm(60000)
  for i in 1:minibatch_size:60000
    minibatch(perm[i+(1:minibatch_size)-1])
  end
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
    end
end

if args["test"]
  testrun,testdict = instance(pool, a)

  ftest = map(T, testdata()[1]/255)
  ltest = map(T, testdata()[2])

  pp = zeros(10, 10000)

  for i in 1:div(10000,minibatch_size)
    n += 1
    ix = (i-1)*minibatch_size+(1:minibatch_size)
    copy!(testdict[x], vec(ftest[:,ix]))
    testrun()
    pp[:,ix] = get_data(testdict[a])
  end

  correct = countnz([indmax(pp[:,i]) == (1+ltest[i]) for i in 1:10000])/10000
  @show correct

  gmean = exp(mean([log(pp[round(Int, 1+ltest[i]), i]) for i in 1:10000]))
  @show gmean
end

if args["store"] !== nothing
  save_snapshot(args["store"], e)
end

if gpu
  free(pool.pool)
  CUDArt.close(0)
end

@show E[end]
