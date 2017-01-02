# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
abstract StoragePool{T}

atype{T}(::StoragePool{T}) = T
Base.eltype{T}(::StoragePool{T}) = eltype(T)

immutable NullPool{T} <: StoragePool{T}
  token::Vector{Symbol}
  arrays::Vector{T}
end

NullPool(T::Type) = NullPool{T}([:nothing], T[])

allocate{T}(n::NullPool{T}, siz::Int...) = let a = T(siz...); push!(n.arrays, a); a end
deallocate{T}(::NullPool{T}, ::T) = nothing

Base.close{T<:Array}(p::NullPool{T}) = empty!(p.arrays)
@cuda function Base.close{T<:CudaArray}(p::NullPool{T})
  for a in p.arrays
    free(a)
  end
  empty!(p.arrays)
end

type PoolCounter
  allocated::Int
  deallocated::Int
  wasted::Int
  unwasted::Int

  maxAllocated::Int
  maxWasted::Int

  PoolCounter() = new(0,0,0,0,0,0)
end

function count_allocated(c::PoolCounter, a::Int, w::Int)
  c.allocated += a
  c.wasted += w
  if c.allocated - c.deallocated > c.maxAllocated
    c.maxAllocated = c.allocated - c.deallocated
    c.maxWasted = c.wasted - c.unwasted
  end
end

function count_deallocated(c::PoolCounter, a::Int, w::Int)
  c.deallocated += a
  c.unwasted += w
end

immutable BuddyPool{T} <: StoragePool{T}
  pool::T
  free::IntSet
  psize::Int
  levels::Int
  map::Dict{Int,Tuple{Int,Int,Int}}
  counter::PoolCounter

  token::Vector{Symbol}

  @compat (::Type{BuddyPool})(pool::Array, N=16) = let NN = min(N, maxlevels(length(pool)))
    init_freelist(new{Array{eltype(pool)}}(pool, IntSet(), div(nextpow2(length(pool)), 1<<(NN-1)), NN, Dict{Int,Int}(), PoolCounter(), [:nothing])) end

  @cuda @compat (::Type{BuddyPool})(pool::CudaArray, N=16) = let NN = min(N, maxlevels(length(pool)))
    init_freelist(new{CudaArray{eltype(pool)}}(pool, IntSet(), div(nextpow2(length(pool)), 1<<(NN-1)), NN, Dict{Int,Int}(), PoolCounter(), [:nothing])) end
end

Base.close{T<:Array}(p::BuddyPool{T}) = nothing
@cuda Base.close{T<:CudaArray}(p::BuddyPool{T}) = free(p.pool)

maxlevels(len) = 64-leading_zeros(UInt64(nextpow2(len)))-4

function init_freelist(b::BuddyPool)
  sizehint!(b.free, 1<<b.levels)
  for i in 1:div(length(b.pool), b.psize)
     deallocate_block(b, (1<<(b.levels-1)) + i - 1)
  end
  b
end

hsize(b::BuddyPool, i::Int) = hsize(sizeof(eltype(b))*i)
hsize(i::Int) = i < 2^10 ? string(i, " B") :
  i < 2^20 ? @sprintf("%4.2f KiB", i/2^10) :
  i < 2^30 ? @sprintf("%4.2f MiB", i/2^20) :
             @sprintf("%4.2f GiB", i/2^30)

counter(b::BuddyPool, i::Int) = "$i ($(hsize(b, i)))"

function print_stats(io::IO, b::BuddyPool)
  len = length(b.pool)
  usable = b.psize*div(len, b.psize)
  println(io, "BuddyPool{$(atype(b))}, $(len) elements ($(hsize(b, len))), $(usable) usable ($(hsize(b, usable)))")
  println(io, "Allocated: $(counter(b, b.counter.allocated)) total, $(counter(b, b.counter.maxAllocated)) max, $(counter(b, b.counter.allocated-b.counter.deallocated)) current")
  println(io, "Wasted: $(counter(b, b.counter.wasted)) total, $(counter(b, b.counter.maxWasted)) max, $(counter(b, b.counter.wasted-b.counter.unwasted)) current")
end

# order 0 is block 1 
# order 1 is blocks 2-3
# order 2 is blocks 4-7 (2^order, 2^(order+1)]

function allocate_block(b::BuddyPool, order::Int)
  i,_ = next(b.free, 1<<order)
  if i < (1<<(order+1))
    delete!(b.free, i)
    return i
  end

  if order == 0
    error("memory pool is exhausted")
  end

  a = allocate_block(b, order-1)<<1
  push!(b.free, a+1)
  return a
end

function deallocate_block(b::BuddyPool, i::Int)
  if i > 1 && ((i$1) in b.free)
    delete!(b.free, i$1)
    deallocate_block(b, i>>1)
  else
    push!(b.free, i)
  end
end

function print_free_list(b::BuddyPool, start::Int, depth::Int)
  for i in 1:depth
    for j in 1:(2^(i-1))
      k = (start<<(i-1)) + j - 1
      local ch
      if k in b.free
        ch = "F"
      else
        ch = "x"
        kk = k
        while kk > 0
          if kk in b.free
            ch = " "
            break
          end
          kk >>= 1
        end
      end
      print(ch^(2^(depth-i)))
    end
    println()
  end
end

function allocate{T}(b::BuddyPool{T}, siz::Int...)
  len = prod(siz)
  len = div(len+b.psize-1, b.psize)
  len = nextpow2(len)
  order = leading_zeros(UInt64(len)) + b.levels - 64
  if order < 0
    error("allocation $siz is larger than the pool")
  end
  bl = allocate_block(b, order)
  count_allocated(b.counter, prod(siz), b.psize*len-prod(siz))
  pa = (((1<<order)-1)&bl) * (1<<(b.levels-order-1))
  b.map[pa] = (bl, prod(siz), len)
  block_to_array(b, pa, siz)
end

function deallocate{T}(b::BuddyPool{T}, p::T)
  pa = div(pointer(p) - pointer(b.pool), b.psize*sizeof(eltype(b.pool)))
  bl,siz,len = b.map[pa]
  delete!(b.map, pa)
  deallocate_block(b, bl)
  count_deallocated(b.counter, siz, b.psize*len-siz)
  nothing
end

block_to_array{T}(b::BuddyPool{Array{T}}, p, siz) = pointer_to_array(pointer(b.pool) + b.psize*p*sizeof(eltype(b.pool)), siz)
@cuda block_to_array{T}(b::BuddyPool{CudaArray{T}}, p, siz) = CudaArray(pointer(b.pool) + b.psize*p*sizeof(eltype(b.pool)), siz, device(b.pool))
