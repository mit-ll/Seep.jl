# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
#__precompile__(true)
module Seep

using HDF5, JLD, Compat

export ANode, sigm, softmax, gradients, evaluator, BuddyPool, NullPool, instance, evaluate,
  gradient_descent, momentum, adadelta, adam, weight_update, graphviz, connected_nodes,
  load_snapshot, save_snapshot, rmsprop, name!, sigmoid_loss, softplus

if !haskey(ENV, "SEEP_NO_GPU")
  using CUDAdrv, CUBLAS, SHA
  import CUDArt: device, free, CudaArray, to_host, device_synchronize
  macro llvm(x) esc(x) end
  macro cuda(x) esc(x) end
  include("llvm/LLVM.jl")
  include("llvm/NVVM.jl")
  import .LLVM: BuilderRef, ValueRef, ModuleRef, TypeRef
  include("cudahacks.jl")
else
  macro cuda(x) end
  macro cuda_kernel(x...) end
  macro cuda_gsl(x...) end
  macro llvm(x) end
end

macro cuda_text(x) end
macro cuda_include(x) end

include("sources.jl")

@llvm include("llvm/llvm-instance.jl")

include("flows.jl")
export Flow, Linear, RNN, LSTM, GRU, Conv2D, Pool, Dropout, RandomNoise
export get_params, reg_params, scanl, scanl!, in_size, out_size

if isdefined(Seep, :do_forward)
  error("I think you meant do_forward!")
end

end
