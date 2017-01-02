# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--gpu"
        help = "Use GPU (cuda)"
        action = :store_true
    "--llvm"
        help = "Use LLVM to gernerate GPU code"
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
    "--dropout"
        help = "Use dropout"
        action = :store_true
    "--once"
        help = "Don't compile the evaluator"
        action = :store_true
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
const llvm = args["llvm"]
const profile = args["profile"]
const minibatch_size = args["minibatch"]
const epochs = args["epochs"]
const dropout = args["dropout"]
const once = args["once"]
const solver = args["solver"]

if !gpu
end

const T = args["64"] ? Float64 : Float32

if gpu
    using CUDArt
    CUDArt.init(0)
    device(0)

    using CUBLAS, MNIST, Seep, JLD

    make_node(n, a::Array) = ANode(n, CudaArray(map(T, a)))
    if once
      make_node(n, x::Int...) = ANode(n, CudaArray(T, x...))
    else
      make_node(n, x::Int...) = ANode(n, x...)
    end
    get_data(x) = to_host(x)
    const pool = BuddyPool(CudaArray(T, 1<<23))
else
    ENV["SEEP_NO_GPU"]=1
    using MNIST, Seep, JLD

    make_node(n, a::Array) = ANode(n, map(T, a))
    if once
      make_node(n, x::Int...) = ANode(n, Array(T, x...))
    else
      make_node(n, x::Int...) = ANode(n, x...)
    end
    get_data(x) = x
    const pool = BuddyPool(Array(T, 1<<23))
end

if args["seed"] != nothing
  srand(args["seed"])
end

const ftrain = map(T, traindata()[1]/255)
const ltrain = map(T, traindata()[2])
const ytrain = zeros(T, 10, length(ltrain))
for i in 1:size(ytrain, 2)
  ytrain[1+round(Int, ltrain[i]),i] = 1
end

const n0 = 28*28
const x = ANode("x", n0, minibatch_size)
a = x

reg_terms = ANode[]
params = ANode[]

if true
  const n1 = 25
  const w1 = make_node("w1", randn(n1, n0)/sqrt(n0))
  const b1 = make_node("b1", zeros(n1, 1))
  a = tanh(w1*a.+b1)

  if dropout
    const d = make_node("d1", zeros(n1, minibatch_size))
    a = a.*d
  end
  push!(params, w1)
  push!(params, b1)
  push!(reg_terms, w1)
else
  const n1 = n0
end

if false
  const w2 = make_node("w2", randn(n2, n1)/sqrt(n1))
  const b2 = make_node("b2", zeros(n2, 1))
  a = tanh(w2*a.+b2)
  push!(params, w2)
  push!(params, b2)
  push!(reg_terms, w2)
else
  const n2 = n1
end

const n3 = 10
const w3 = make_node("w3", randn(n3, n2)/sqrt(n2))
const b3 = make_node("b3", zeros(n3, 1))
a = softmax(w3*a.+b3, 1)
push!(params, w3)
push!(params, b3)
push!(reg_terms, w3)

const y = ANode("y", 10, minibatch_size)
const e = -log(a).*y

const obj= sum(e) + 1e-3*reduce(+, map(w->dot(w,w), reg_terms))

const grad = gradients(obj)

if solver == "sgd"
  const sol = gradient_descent
elseif solver == "momentum"
  const sol = momentum
elseif solver == "adadelta"
  const sol = Seep.adadelta
elseif solver == "adam"
  const sol = adam
elseif solver == "adamfast"
  const sol = Seep.adam_fast
else
  error("Unknown solver: $solver")
end

upd = map(x->sol(x, grad[x]/sqrt(minibatch_size)), params)

const E = Float64[]
const times = Float64[]

if args["graphviz"]
  open(f->graphviz(f,  e, upd...), "mnist.gv", "w")
end

if args["devectorize"]
  Seep.devectorize(e, upd...)
end

if once
  run() = global dict = evaluate(pool, e, a, upd...)
else
  const run,dict = (gpu && llvm ? Seep.llvm_instance : instance)(pool, ANode[e; a; upd])
end

Seep.print_stats(STDOUT, pool)

if args["load"] !== nothing
  load_snapshot(args["load"], e)
end

n = 0

if once
  const xa,ya = x.arg,y.arg
else
  const xa,ya = dict[x],dict[y]
end


if dropout && once
  const da = d.arg
elseif dropout
  const da = dict[d]
end

function minibatch(ii)
    if gpu
       copy!(xa, ftrain[:,ii])
       copy!(ya, ytrain[:,ii])
       if dropout
         copy!(da, 2. * bitrand(n1,length(ii)))
       end
    else
        for i in 1:length(ii)
          for j in 1:(28*28)
            xa[j,i] = ftrain[j,ii[i]]
          end

          for j in 1:10
            ya[j,i] = ytrain[j,ii[i]]
          end

          if dropout
            for j in 1:n1
              da[j,i] = 2. * rand(0:1)
            end
          end
        end
    end

    run()

    push!(E, sum(get_data(dict[e]))/length(ii))
    push!(times, time())
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
  ftest = map(T, testdata()[1]/255)
  ltest = map(T, testdata()[2])

  pp = zeros(10, 10000)
  if dropout
    da[:] = ones(n1, minibatch_size)
  end

  for i in 1:div(10000,minibatch_size)
    n += 1
    ix = (i-1)*minibatch_size+(1:minibatch_size)
    copy!(xa, vec(ftest[:,ix]))
    run()
    pp[:,ix] = get_data(dict[a])
  end

  correct = countnz([indmax(pp[:,i]) == (1+ltest[i]) for i in 1:10000])/10000
  @show correct

  gmean = exp(mean([log(pp[round(Int, 1+ltest[i]), i]) for i in 1:10000]))
  @show gmean
end

if args["store"] !== nothing
  save_snapshot(args["store"], e)
end

close(pool)

if gpu
  CUDArt.close(0)
end

@show E[end]
