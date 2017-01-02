using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--epochs"
        help = "Epochs to train"
        arg_type = Int
        default = 100
    "--save-snapshot"
        help = "Whether to Save Snapshots"
        arg_type = Bool
        default = true
    "--snapshot-dir"
        help = "Directory to Store Snapshots"
        arg_type = ASCIIString
        default = "snapshot-brnn"
    "--snapshot_n_iter"
        help = "How many iterations between snapshots"
        arg_type = Int
        default = 10
    "--stat_n_iter"
        help = "How many iterations between saving statistics"
        arg_type = Int
        default = 10
end
const args = parse_args(ARGS, s)


ENV["SEEP_NO_GPU"]=false
using Seep, JLD
T = Float32
relu(x::ANode) = Seep.max!(0, x)
make_node(a::Array) = ANode(a)
make_node(x::Int...) = ANode(x...)
get_data(x) = x
const pool = BuddyPool(Array{T}(1<<24))

include("data.jl")
data = IRData()

# Constants
const sequence_length = 50
const input_size = 2
const hidden_size = [10, 10]
const feature_size = 5
const output_size = 2
const batch_size = 80

# Network
const X = ANode[make_node(input_size, batch_size) for i=1:sequence_length]
const Y = make_node(output_size, batch_size)
y = make_node(zeros(T, output_size, batch_size))

params = Flow[]
let 
    # Initialize nodes
    fe = Linear(T, "e", input_size, hidden_size[1])
    fhf = Flow[]
    fhb = Flow[]
    for j = 1:length(hidden_size)
        k = j==1 ? 1 : j-1

        f = LSTM(T, "h$(j)_f", hidden_size[k], hidden_size[j])
        push!(fhf, f)

        f = LSTM(T, "h$(j)_b", hidden_size[k], hidden_size[j])
        push!(fhb, f)
    end
    fxf = Linear(T, "xf", hidden_size[end], feature_size)
    fxb = Linear(T, "xb", hidden_size[end], feature_size)
    ff = Linear(T, "features", feature_size, feature_size)
    fy = Linear(T, "output", feature_size, output_size)
    push!(params, fe, fhf..., fhb..., fxf, fxb, ff, fy)

    # Initialize LSTM variables
    hf = ANode[make_node(zeros(T, hidden_size[i], batch_size)) for i = 1:length(hidden_size)]
    cf = ANode[make_node(zeros(T, hidden_size[i], batch_size)) for i = 1:length(hidden_size)]
    hb = ANode[make_node(zeros(T, hidden_size[i], batch_size)) for i = 1:length(hidden_size)]
    cb = ANode[make_node(zeros(T, hidden_size[i], batch_size)) for i = 1:length(hidden_size)]

    # Build graph
    xf = make_node(zeros(T, input_size, batch_size))
    xb = make_node(zeros(T, input_size, batch_size))
    for i = 1:length(X)
        xf = relu(fe(X[i]))
        xb = relu(fe(X[length(X)-i+1]))

        for j = 1:length(hidden_size)
            cf[j], hf[j] = fhf[j](xf, cf[j], hf[j])
            xf = hf[j]

            cb[j], hb[j] = fhb[j](xb, cb[j], hb[j])
            xb = hb[j]
        end
    end

    xf = fxf(xf)
    xb = fxb(xb)
    x = tanh(ff(xf + xb))
    y = softmax(fy(x), 1)
end

# Loss
const loss = -log(y) .* Y
const obj = sum(loss)
const grad = gradients(obj)

# Gradient Calculation for SGD
updates = map(x->Seep.adam_fast(x, grad[x]), get_params(params))
const BRNN = instance(pool, y, loss, updates...)
Seep.print_stats(STDOUT, pool)

# Load Latest Snapshot
n_epoch_start = 0

if args["save-snapshot"]
    if isdir(args["snapshot-dir"])
        f = filter(x->contains(x, "snapshot-"), readdir(args["snapshot-dir"]))
        if length(f) > 0
            epochs = map(x->parse(Int, split(split(x, ".")[1], "-")[end]), f)
            ind = indmax(epochs)
            n_epoch_start = epochs[ind]
            @show n_epoch_start
            load_snapshot(joinpath(args["snapshot-dir"], f[ind]), loss)
        end
    else
        mkdir(args["snapshot-dir"])
    end
end

# Train
function epoch()
    n = data.n_epochs
    while data.n_epochs == n
        xx, yy = next_batch(data, batch_size)

        for i = 1:length(X)
            copy!(BRNN[X[i]], xx[:, i, :])
        end

        ytrain = zeros(T, output_size, batch_size)
        for i in 1:batch_size
          ytrain[round(Int, yy[i]),i] = 1
        end

        copy!(BRNN[Y], ytrain)

        BRNN()
    end
end

E = zeros(args["stat_n_iter"])
for i = 1:args["epochs"]
    epoch()

    l = sum(get_data(BRNN[loss]))/batch_size
    println("Loss: ", l)

    E[mod(i-1, args["stat_n_iter"])+1] = l
    if mod(i, args["stat_n_iter"])==0
        f = joinpath(args["snapshot-dir"], "statistics.jld")
        if isfile(f)
            E = vcat(load(f, "E"), E)
        end
        save(f, "E", E)
        E = zeros(args["stat_n_iter"])
    end


    if mod(i, args["snapshot_n_iter"])==0 && args["save-snapshot"]
        save_snapshot(joinpath(args["snapshot-dir"], "snapshot-$(i+n_epoch_start).jld"), loss)
    end
end

