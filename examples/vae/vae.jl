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
        default = "snapshot-vae"
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
using MNIST, Seep, JLD
T = Float64
const pool = BuddyPool(Array{T}(1<<24))
const ftrain = map(T, traindata()[1]/255)
import Seep: @named


# Constants
const input_size = 28*28
const latent_size = 5
const output_size = input_size
const batch_size = 100

# Network
X = ANode("X", zeros(T, input_size, batch_size))
μ = ANode("μ", zeros(T, latent_size, batch_size))
lnσ = ANode("lnσ", zeros(T, latent_size, batch_size))

x = X

# Generate latent variables (encoding)
encode_params = Flow[]
let encode_size=[20, 20]

    fe = Linear(T, "e_encode", input_size, encode_size[1])
    x = tanh(fe(x))

    fh = Flow[]
    for i in 1:length(encode_size)
        j = i==1 ? i : i-1
        fi = Linear(T, "h_$(i)_encode", encode_size[j], encode_size[i])
        x = tanh(fi(x))
        push!(fh, fi)
    end
    fs = Linear(T, "s_encode", encode_size[end], latent_size)
    x = tanh(fs(x))

    fμ = Linear(T, "μ_encode", latent_size, latent_size)
    μ = fμ(x)

    fσ = Linear(T, "lnσ_encode", latent_size, latent_size)
    lnσ = fσ(x)

    push!(encode_params, fe, fh..., fs, fμ, fσ)
end

# Sample Latent Variables
@named ϵ = randn!(ANode(zeros(T, latent_size, batch_size)))
@named z = μ + exp(lnσ) .* ϵ

# Inference of image from latent variables (decode)
yhat = z
decode_params = Flow[]
let decode_size=[20, 20]

    fe = Linear(T, "e_decode", latent_size, decode_size[1])
    yhat = tanh(fe(yhat))

    fh = Flow[]
    for i in 1:length(decode_size)
        j = i==1 ? i : i-1
        fi = Linear(T, "h_$(i)_decode", decode_size[j], decode_size[i])
        yhat = tanh(fi(yhat))
        push!(fh, fi)
    end
    fs = Linear(T, "s_decode", decode_size[end], output_size)
    yhat = tanh(fs(yhat))

    fy = Linear(T, "y_decode", output_size, output_size)
    yhat = fy(yhat)

    push!(decode_params, fe, fh..., fs, fy)
end

# Loss
@named begin
  kl   = (0.5 / latent_size) * sum(μ.^2 + exp(lnσ) - lnσ - 1, 1)
  loss = 0.5*sum((X - yhat).^2, 1)
  obj  = sum(loss) + sum(kl)
end

grad = gradients(obj)

# Gradient Calculation for SGD
updates = map(x->Seep.adam(x, grad[x]), get_params(Flow[encode_params; decode_params]))
const VAE = instance(pool, [X; yhat; kl; loss; updates])
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
    perm = randperm(60000)
    for i in 1:batch_size:60000
        ii = perm[i+(1:batch_size)-1]

        # Input Image
        VAE.X[:,:] = ftrain[:, ii]
        
        VAE()
    end
end

Erec = zeros(args["stat_n_iter"])
Ekl = zeros(args["stat_n_iter"])
for i = 1:args["epochs"]
    epoch()

    rec_loss = sqrt(sum(VAE.loss))/batch_size
    kl_loss = sum(VAE.kl)/batch_size
    println("Rec Loss: ", rec_loss, " KL Loss:", kl_loss)

    Erec[mod(i-1, args["stat_n_iter"])+1] = rec_loss
    Ekl[mod(i-1, args["stat_n_iter"])+1] = kl_loss
    if mod(i, args["stat_n_iter"])==0
        f = joinpath(args["snapshot-dir"], "statistics.jld")
        if isfile(f)
            Erec = vcat(load(f, "Erec"), Erec)
            Ekl = vcat(load(f, "Ekl"), Ekl)
        end
        save(f, "Erec", Erec, "Ekl", Ekl)
        Erec = zeros(args["stat_n_iter"])
        Ekl = zeros(args["stat_n_iter"])
    end


    if mod(i, args["snapshot_n_iter"])==0 && args["save-snapshot"]
        save_snapshot(joinpath(args["snapshot-dir"], "snapshot-$(i+n_epoch_start).jld"), loss)
    end
end

