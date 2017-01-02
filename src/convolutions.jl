# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using Base.Cartesian

pad0(N::Int, t::Tuple{Vararg{Int}}) = (t..., zeros(Int, N-length(t))...)::NTuple{N,Int}
pad1(N::Int, t::Tuple{Vararg{Int}}) = (t..., ones(Int, N-length(t))...)::NTuple{N,Int}

@generated function im2col{N}(fsize::NTuple{N,Int}, xsize::NTuple{N,Int}, stride::NTuple{N,Int})
  quote
    n = 0
    c = zeros(Int, prod(fsize), 1)
    @nloops $N i j->1:fsize[j] begin
      n += 1
      @nexprs $N j->c[n] += (i_j-1)*prod(xsize[1:(j-1)])
    end

    outsize = ([max(0, div(xsize[i]-fsize[i],stride[i])+1) for i in 1:$N]...)
    ind = ones(Int, 1, prod(outsize))
    n = 0
    @nloops $N i k->1:outsize[k] begin
      n += 1
      @nexprs $N j->ind[n] += (i_j-1)*stride[j]*prod(xsize[1:(j-1)])
    end

    ind.+c
  end
end

macro ind(exp, i, fun)
  @assert exp.head == :ref
  append!(exp.args, @eval map($fun, $i))
  exp
end

function ind(exp, i, fun)
  @assert exp.head == :ref
  append!(exp.args, map(fun, i))
  exp
end

@cuda_include "cudaConv.cu"

# Direct Convolution

immutable ConvOpt{N}
  xsize::NTuple{N,Int}
  fsize::NTuple{N,Int}
  stride::NTuple{N,Int}
  padding::NTuple{N,Int}

  @compat function (::Type{ConvOpt})(fsize::Tuple{Vararg{Int}}, xsize::Tuple{Vararg{Int}},
    stride::Tuple{Vararg{Int}}, padding::Tuple{Vararg{Int}})
    NN = length(xsize)
    @assert length(fsize) <= NN
    @assert length(stride) <= NN
    @assert length(padding) <= NN
    new{NN}(xsize, pad1(NN, fsize), pad1(NN, stride), pad0(NN, padding))
  end
end

function outsize{N}(o::ConvOpt{N}, numout::Int=1)
  ([(div(2*o.padding[i]+o.xsize[i]-o.fsize[i],o.stride[i])+1)*(i > 1 ? 1 : numout) for i in 1:N]...)
end

function Base.conv2(filter::ANode, x::ANode;
  stride=(ones(Int, ndims(x))...), padding=(zeros(Int, ndims(x))...))
  opt = ConvOpt(size(filter)[2:end], size(x), stride, padding)
  ANode(:conv2, (filter, x), outsize(opt, size(filter, 1)), opt)
end

gradient_node(n::ANode{:conv2}, ::Type{Val{1}}, b) = ANode(:conv2_bp_filter, (n.input[2], b), size(n.input[1]), n.arg)
gradient_node(n::ANode{:conv2}, ::Type{Val{2}}, b) = ANode(:conv2_bp_data, (n.input[1], b), size(n.input[2]), n.arg)

@generated function do_forward!{T,N}(n::ANode{:conv2}, y::Array{T,N}, filter::Array{T}, x::Array{T,N})
  quote
    s = n.arg.stride
    p = n.arg.padding
    fill!(y, zero(T))
    @fastmath @inbounds begin
      @nloops $N i d->1:(div(size(x,d) - size(filter,d+1) + 2p[d], s[d]) + 1) begin
        @nloops $N j d->1:size(filter, d+1) begin
          xi = (@nref $N x d->clamp((i_d-1)*s[d]+j_d-p[d], 1, size(x, d)))
          @simd for j_0 in 1:size(filter, 1)
            $(ind(:(y[j_0 + (i_1-1)*size(filter, 1)]), 2:N, d->Symbol("i_",d))) += $(ind(:(filter[]), 0:N, d->Symbol("j_", d))) * xi
          end
        end
      end
    end
  end
end

@generated function do_forward!{T,N}(n::ANode{:conv2_bp_filter}, y::Array{T}, x::Array{T,N}, b::Array{T,N})
  quote
    s = n.arg.stride
    p = n.arg.padding
    fill!(y, zero(T))
    @fastmath @inbounds begin
      @nloops $N i d->1:(div(size(x,d) - size(y,d+1) + 2p[d], s[d]) + 1) begin
        @nloops $N j d->1:size(y, d+1) begin
          xi = (@nref $N x d->clamp((i_d-1)*s[d]+j_d-p[d], 1, size(x, d)))
          @simd for j_0 in 1:size(y, 1)
            bi = $(ind(:(b[j_0 + (i_1-1)*size(y, 1)]), 2:N, d->Symbol("i_",d)))
            $(ind(:(y[]), 0:N, d->Symbol("j_", d))) += bi*xi
          end
        end
      end
    end
  end
end

@generated function do_forward!{T,N}(n::ANode{:conv2_bp_data}, y::Array{T,N}, filter::Array{T}, b::Array{T,N})
  quote
    s = n.arg.stride
    p = n.arg.padding
    fill!(y, zero(T))
    @fastmath @inbounds begin
      @nloops $N i d->1:(div(size(y,d) - size(filter,d+1) + 2p[d], s[d]) + 1) begin
        @nloops $N j d->1:size(filter, d+1) begin
          @simd for j_0 in 1:size(filter, 1)
            bi = $(ind(:(b[j_0 + (i_1-1)*size(filter, 1)]), 2:N, d->Symbol("i_",d)))
            fi = $(ind(:(filter[]), 0:N, d->Symbol("j_", d)))
            (@nref $N y d->clamp((i_d-1)*s[d]+j_d-p[d], 1, size(y, d))) += bi*fi
          end
        end
      end
    end
  end
end

@cuda begin
  const conv3_Float32 = CuFunction(md, "conv3_Float32")
  const conv3_Float64 = CuFunction(md, "conv3_Float64")
  const conv3_bp_data_Float32 = CuFunction(md, "conv3_bp_data_Float32")
  const conv3_bp_data_Float64 = CuFunction(md, "conv3_bp_data_Float64")
  const conv3_bp_filter_Float32 = CuFunction(md, "conv3_bp_filter_Float32")
  const conv3_bp_filter_Float64 = CuFunction(md, "conv3_bp_filter_Float64")

  @generated function do_forward!{T}(n::ANode{:conv2}, y::CudaArray{T,4}, filter::CudaArray{T,4}, x::CudaArray{T,4})
     quote
       o = outsize(n.arg, size(filter,1))
       bs = (8,8,8)
       gs = ((o[1]+bs[1]-1)÷bs[1], (o[2]+bs[2]-1)÷bs[2], o[4]*((o[3]+bs[3]-1)÷bs[3]))
       opt = CudaArray(UInt8, sizeof(n.arg))
       copy!(opt, reinterpret(UInt8, [n.arg]))
       shmem = (bs[1]*n.arg.stride[1]+n.arg.fsize[1])*(bs[2]*n.arg.stride[2]+n.arg.fsize[2])*(bs[3]*n.arg.stride[3]+n.arg.fsize[3])
       cudacall($(Symbol("conv3_", T)), gs, bs, (Ptr, Int, Ptr{T}, Ptr{T}, Ptr{T}), opt.ptr, size(filter, 1), y.ptr, filter.ptr, x.ptr; shmem_bytes=sizeof(T)*shmem)
       free(opt)
     end
  end

  @generated function do_forward!{T}(n::ANode{:conv2_bp_data}, dx::CudaArray{T,4}, f::CudaArray{T,4}, b::CudaArray{T,4})
     quote
       o = size(dx)
       bs = (8,8,8)
       gs = ((o[1]+bs[1]-1)÷bs[1], (o[2]+bs[2]-1)÷bs[2], o[4]*((o[3]+bs[3]-1)÷bs[3]))
       opt = CudaArray(UInt8, sizeof(n.arg))
       copy!(opt, reinterpret(UInt8, [n.arg]))
       cudacall($(Symbol("conv3_bp_data_", T)), gs, bs, (Ptr, Int, Ptr{T}, Ptr{T}, Ptr{T}), opt.ptr, size(f, 1), dx.ptr, f.ptr, b.ptr)
       free(opt)
     end
  end

  @generated function do_forward!{T}(n::ANode{:conv2_bp_filter}, df::CudaArray{T,4}, x::CudaArray{T,4}, b::CudaArray{T,4})
     quote
       o = size(df)
       bs = (8,8,8)
       gs = ((o[1]*o[2]+bs[1]-1)÷bs[1], (o[3]+bs[2]-1)÷bs[2], (o[4]+bs[3]-1)÷bs[3])
       opt = CudaArray(UInt8, sizeof(n.arg))
       copy!(opt, reinterpret(UInt8, [n.arg]))
       shmem = (bs[1]*n.arg.stride[1]+n.arg.fsize[1])*(bs[2]*n.arg.stride[2]+n.arg.fsize[2])*(bs[3]*n.arg.stride[3]+n.arg.fsize[3])
       cudacall($(Symbol("conv3_bp_filter_", T)), gs, bs, (Ptr, Int, Ptr{T}, Ptr{T}, Ptr{T}), opt.ptr, size(df, 1), df.ptr, x.ptr, b.ptr; shmem_bytes=sizeof(T)*shmem)
       free(opt)
     end
  end
end

# Pooling

function pool(x::ANode, sz::Tuple{Vararg{Int}}; stride=sz, padding=(0,))
  opt = ConvOpt(sz, size(x), stride, padding)
  ANode(:pool, (x,), outsize(opt), opt)
end

gradient_node(n::ANode{:pool}, ::Type{Val{1}}, b) = ANode(:pool_bp, (n.input[1], n, b), size(n.input[1]), n.arg)

@generated function do_forward!{T,N}(n::ANode{:pool}, y::Array{T,N}, x::Array{T,N})
  quote
    s = n.arg.stride
    p = n.arg.padding
    sz = n.arg.fsize
    @fastmath @inbounds begin
      @nloops $N i d->1:(div(size(x,d) - sz[d] + 2p[d], s[d]) + 1) begin
        a = @nref $N x d->clamp((i_d-1)*s[d]+1, 1, size(x, d))
        @nloops $N j d->1:sz[d] begin
          b = @nref $N x d->clamp((i_d-1)*s[d]+j_d-p[d], 1, size(x, d))
          if b > a
            a = b
          end
        end
        (@nref $N y d->i_d) = a
      end
    end
  end
end

@generated function do_forward!{T,N}(n::ANode{:pool_bp}, y::Array{T,N}, x::Array{T,N}, z::Array{T,N}, b::Array{T,N})
  quote
    s = n.arg.stride
    p = n.arg.padding
    sz = n.arg.fsize
    fill!(y, zero(T))
    @fastmath @inbounds begin
      @nloops $N i d->1:(div(size(x,d) - sz[d] + 2p[d], s[d]) + 1) begin
        zi = (@nref $N z d->i_d)
        @nloops $N j d->1:sz[d] begin
          if (@nref $N x d->clamp(((i_d-1)*s[d]+j_d), 1, size(x, d))) == zi
            (@nref $N y d->clamp((i_d-1)*s[d]+j_d, 1, size(y, d))) += (@nref $N b d->i_d)
          end
        end
      end
    end
  end
end

@cuda begin
  const pool_Float32 = CuFunction(md, "pool_Float32")
  const pool_Float64 = CuFunction(md, "pool_Float64")
  const pool_bp_Float32 = CuFunction(md, "pool_bp_Float32")
  const pool_bp_Float64 = CuFunction(md, "pool_bp_Float64")

  @generated function do_forward!{T}(n::ANode{:pool}, y::CudaArray{T,4}, x::CudaArray{T,4})
     quote
       o = size(y)
       bs = (8,8,8)
       gs = ((o[1]+bs[1]-1)÷bs[1], (o[2]+bs[2]-1)÷bs[2], o[4]*((o[3]+bs[3]-1)÷bs[3]))
       opt = CudaArray(UInt8, sizeof(n.arg))
       copy!(opt, reinterpret(UInt8, [n.arg]))
       shmem = (bs[1]*n.arg.stride[1]+n.arg.fsize[1])*(bs[2]*n.arg.stride[2]+n.arg.fsize[2])*(bs[3]*n.arg.stride[3]+n.arg.fsize[3])
       cudacall($(Symbol("pool_", T)), gs, bs, (Ptr, Ptr{T}, Ptr{T}), opt.ptr, y.ptr, x.ptr; shmem_bytes=sizeof(T)*shmem)
       free(opt)
     end
  end

  @generated function do_forward!{T}(n::ANode{:pool_bp}, dx::CudaArray{T,4}, x::CudaArray{T,4}, y::CudaArray{T,4}, b::CudaArray{T,4})
     quote
       fill!(dx, zero(eltype(dx)))
       o = size(dx)
       bs = (8,8,8)
       gs = ((o[1]+bs[1]-1)÷bs[1], (o[2]+bs[2]-1)÷bs[2], o[4]*((o[3]+bs[3]-1)÷bs[3]))
       opt = CudaArray(UInt8, sizeof(n.arg))
       copy!(opt, reinterpret(UInt8, [n.arg]))
       cudacall($(Symbol("pool_bp_", T)), gs, bs, (Ptr, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{T}), opt.ptr, dx.ptr, x.ptr, y.ptr, b.ptr)
       free(opt)
     end
  end
end
