# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
# An unreasonably effective Recurrent Neural Network.
#
# based on http://karpathy.github.io/2015/05/21/rnn-effectiveness/ (and
# https://github.com/karpathy/char-rnn).

ENV["SEEP_NO_GPU"]="1"
using Seep
import Seep: relu, @named

#
# Content to reproduce
#
const path = ENV["HOME"] * "/projects/julia/base"
const files = map(utf8, map(readall, filter(x->endswith(x, ".jl")&&!endswith(x, "_symbols.jl"), map(x->joinpath(path, x), readdir(path)))))
#const files = UTF8String["abcdefg"]

@show reduce(+, map(length, files))
const chars = (sort!(unique(reduce(vcat, map(unique, files)))));
@show length(chars)

#
# Network Parameters
#
const btt = 25
const nh = 512
const nl = length(chars)+1
const minibatch = 10

const xs = Seep.ANode[Seep.ANode("x$i", zeros(Float32, length(chars)+1, minibatch)) for i in 1:btt+1]

const layers = Seep.Flow[
    Seep.LSTM(Float32, "1", nl, nh),
    Seep.LSTM(Float32, "2", nh, nl),
    Seep.Linear(Float32, "3", nl, nl),
    x->Seep.softmax(x, 1)
]

#
# Build Network
#

#hs = layers(xs); doesn't work because I need access to the hidden cells
hs = xs[1:btt]

@named c0_1 Seep.ANode(zeros(Float32, nh, minibatch))
@named h0_1 Seep.ANode(zeros(Float32, nh, minibatch))
ch_1 = Seep.apply_lstm(layers[1], hs, (c0_1, h0_1))

@named c1_1 ch_1[1][1]
@named h1_1 ch_1[1][2]
@named cend_1 ch_1[end][1]
@named hend_1 ch_1[end][2]

hs = map(x->x[2], ch_1)

@named c0_2 Seep.ANode(zeros(Float32, nl, minibatch))
@named h0_2 Seep.ANode(zeros(Float32, nl, minibatch))
ch_2  = Seep.apply_lstm(layers[2], hs, (c0_2, h0_2))

@named c1_2 ch_2[1][1]
@named h1_2 ch_2[1][2]
@named cend_2 ch_2[end][1]
@named hend_2 ch_2[end][2]

hs = map(x->x[2], ch_2)

hs = layers[3:4](hs)

@named y1 hs[1]

#
# Optimizer/sampler
#

@named loss reduce(+, [sum(xs[i+1] .* -log(hs[i]+(1-xs[i+1]))) for i in 1:btt])
@named reg reduce(+, [sum(θ.^2) for θ in Seep.reg_params(layers)])

∇ = Seep.gradients(loss + 1e-6*reg)
updates = map(θ->Seep.adam_fast(θ, ∇[θ]), Seep.get_params(layers))

pool = Seep.BuddyPool(Array{Float32}(1<<30))

const g = Seep.instance(pool, Seep.ANode[loss; reg; xs; hs; updates;
    c0_1; h0_1; c0_2; h0_2; cend_1; hend_1; cend_2; hend_2]);

const gt = Seep.instance(pool, Seep.ANode[xs[1]; hs[1];
    c0_1; h0_1; c0_2; h0_2; c1_1; h1_1; c1_2; h1_2]);

#
# Populate xs (network input)
#

const fileno = rand(1:length(files), minibatch)
const index = zeros(Int, minibatch)

function advance()
    for j in 1:minibatch
        if index[j] <= endof(files[fileno[j]])
            g.c0_1[:,j] = g.cend_1[:,j]
            g.h0_1[:,j] = g.hend_1[:,j]
            g.c0_2[:,j] = g.cend_2[:,j]
            g.h0_2[:,j] = g.hend_2[:,j]
        else
            index[j] = 0
            fileno[j] = rand(1:length(files))
            g.c0_1[:,j] = 0
            g.h0_1[:,j] = 0
            g.c0_2[:,j] = 0
            g.h0_2[:,j] = 0
        end
    end
    
    for i in 1:btt+1, j in 1:minibatch
        for k in 1:length(chars)+1
            g[xs[i]][k,j] = 0
        end

        if index[j] == 0
            ;
        elseif index[j] <= endof(files[fileno[j]])
            a = searchsortedfirst(chars, files[fileno[j]][index[j]])
            g[xs[i]][a,j] = 1
        else
            g[xs[i]][end,j] = 1
        end
        if i <= btt
          index[j] = nextind(files[fileno[j]], index[j])
        end
    end
end

#
# Sample a few characters of from the network
#

function sample()
    st = sizehint!(Char[], 100)
    gt.c0_1[:] = 0
    gt.h0_1[:] = 0
    gt.c0_2[:] = 0
    gt.h0_2[:] = 0
    gt[xs[1]][:] = 0

    # for i in files[1][1:10]
    #     gt()
    #     st *= string(i)
    #     a = findfirst(x->x==i, chars)
    #     gt[c0_1][:] = gt[c1_1]
    #     gt[h0_1][:] = gt[h1_1]
    #     gt[c0_2][:] = gt[c1_2]
    #     gt[h0_2][:] = gt[h1_2]
    #     gt[xs[1]][:,1] = 0
    #     gt[xs[1]][a,1] = 1
    # end

    for i in 1:100
        gt()

        p = gt.y1[:,1]
        if i == 1
            p[end] = 0
        end
        #a = indmax(p)
        r = rand()
        cumsum!(p, p)
        p[:] /= p[end]
        a = findfirst(p .>= r)
        if a > length(chars)
            push!(st, '\004')
            break
        elseif p[a] == 0
            push!(st, '∅')
        else
            push!(st, chars[a])
        end

        gt.c0_1[:] = gt.c1_1
        gt.h0_1[:] = gt.h1_1
        gt.c0_2[:] = gt.c1_2
        gt.h0_2[:] = gt.h1_2
        gt[xs[1]][:,1] = 0
        gt[xs[1]][a,1] = 1
    end

    return utf8(st)
end

#
# The loop
#

i = 0
while true
    i += 1
    index[:] = typemax(Int)

    avgloss = 0.

    println("======= $i ========")
    @time for j in 1:100
        advance()
        g()
        avgloss += g[loss][1]
    end

    avgloss /= minibatch*btt*100
    println("loss: $avgloss gmean: 1/$(exp(avgloss)) reg: $(g[reg][1])")
    if isnan(avgloss) || isnan(g[reg][1])
        error("NaN")
    end

    println(@time sample())
    println()

    Seep.save_snapshot("sentence.jld", y1)
end
