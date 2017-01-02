# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
abstract Instance
Base.keys(i::Instance) = keys(i._storage)
Base.values(i::Instance) = values(i._storage)
Base.getindex(i::Instance, a::ANode) = i._storage[a]

@deprecate evaluator(x...) instance(x...)

instance(n::ANode...) = instance(collect(ANode, n))
instance(T::Type, n::ANode...) = instance(T, collect(ANode, n))
instance(pool::StoragePool, n::ANode...) = instance(pool, collect(ANode, n))

instance{A<:ANode}(n::Vector{A}) = instance(Array{Float64}, n)
instance{A<:ANode}(T::Type, n::Vector{A}) = instance(NullPool(T), n)

output_node(n::ANode) = mutates(n) ? output_node(n.input[1]) : n

type RefCounter
  count::Int
  RefCounter() = new(0)
end

incr(r::RefCounter) = r.count += 1
decr(r::RefCounter) = (@assert r.count > 0; r.count -= 1; r.count == 0)
iszero(r::RefCounter) = r.count == 0

incr{T}(d::Dict{T,RefCounter}, x::T) = incr(get!(RefCounter, d, x))
decr{T}(d::Dict{T,RefCounter}, x::T) = decr(d[x])
iszero{T}(d::Dict{T,RefCounter}, x::T) = iszero(d[x])

function generate_function(sorted::Vector{ANode}, dict)
  b = :(begin end)

  if length(sorted) >= 20
    n = length(sorted)
    for i in 1:8
      ix = div((i-1)*n,8)+1:div(i*n,8)
      f = generate_function(sorted[ix], dict)
      push!(b.args, :($f()::Void))
    end
  else
    for i in sorted
      if sym(i) == :input || sym(i) == :const
        continue
      elseif sym(i) == :load
        if dict[i] != arg(i)
          push!(b.args, :(copy!($(dict[i]), $(arg(i)))))
        end
      else
        in = map(x->dict[x], i.input)
        if mutates(i)
          push!(b.args, :(Seep.do_forward!($i, $(in...))))
        else
          out = dict[i]
          push!(b.args, :(Seep.do_forward!($i, $out, $(in...))))
        end
      end
    end
  end

  push!(b.args, :nothing)

  @gensym SeepFunction
  @eval @compat $SeepFunction() = $b

  return SeepFunction
end

function instance{A<:ANode}(pool::StoragePool, n::Vector{A})
  sorted = toposort(n)

  T = atype(pool)
  dict = Dict{ANode,T}()
  userdict = Dict{ANode,T}()

  refcnt = Dict{ANode,RefCounter}()
  for a in n
    if sym(a) == :noop continue end
    incr(refcnt, output_node(a))
  end

  for a in sorted
    for b in a.input
      incr(refcnt, output_node(b))
    end
  end

  constants = ANode[]
  c_block = :(begin end)
  for i in sorted
    if sym(i) == :input
      userdict[i] = dict[i] = allocate(pool, i.size...)
    elseif sym(i) == :const
      if isa(arg(i), T)
        dict[i] =  arg(i)
      else
        incr(refcnt, i)
        t = dict[i] = allocate(pool, i.size...)
        push!(c_block.args, :(copy!($t, map(eltype($T), $(arg(i))))))
        push!(constants, i)
      end
    end
  end

  @gensym initialize_constants
  @eval $(initialize_constants)() = $c_block

  for i in sorted
    @assert !iszero(refcnt, output_node(i))

    if mutates(i)
      dict[i] = dict[i.input[1]]
    elseif sym(i) == :load && isa(arg(i), T)
      dict[i] = arg(i)
    elseif sym(i) != :input && sym(i) != :const
      dict[i] = allocate(pool, i.size...)
    end

    for a in i.input
      aa = output_node(a)
      if decr(refcnt, aa) && arg(aa) != dict[aa]
        deallocate(pool, dict[aa])
      end
    end
  end

  for i in n
    if sym(i) == :noop continue end
    userdict[i] = dict[i]

    aa = output_node(i)
    if decr(refcnt, aa) && arg(aa) != dict[aa]
      deallocate(pool, dict[aa])
    end
  end

  for i in constants
    decr(refcnt, i)
    deallocate(pool, dict[i])
  end

  for i in keys(refcnt)
    @assert iszero(refcnt, i)
  end

  local b
  if isempty(constants)
    b = :(begin end)
  else
    b = :(begin
      if $(pool).token[1] != $(QuoteNode(initialize_constants))
        $(pool).token[1] = $(QuoteNode(initialize_constants))
        $(initialize_constants)()
      end
    end)
  end

  fun = generate_function(sorted, dict)
  push!(b.args, :($fun()::Void))

  @gensym SeepInstance
  typeblock = quote
    immutable $SeepInstance <: Instance
      _storage::Dict{ANode,$T}
      _pool::StoragePool
    end
    Base.show(io::IO, ::$SeepInstance) = print(io, "Seep Instance with $($(length(sorted))) nodes.")
    @compat (::$SeepInstance)() = $b
    $SeepInstance($userdict, $pool)
  end

  for (n,a) in userdict
    if haskey(userdict, n) && n.name != ""
      if length(filter(x->x.name == n.name, sorted)) > 1
        warn("There are multiple nodes named $(n.name).")
        continue
      end
      push!(typeblock.args[2].args[3].args, :($(symbol(n.name))::$(typeof(a))))
      push!(typeblock.args[end].args, a)
    end
  end

  eval(typeblock)
end

# make sure run,dict = instance(...) works for a while, even though run and dict are the same thing
#Base.start(::Instance) = (Base.depwarn("replace run,dict=instance(...) with inst=instance(...)", :SeepStart); 1)
Base.start(::Instance) = 1
Base.done(::Instance, i::Int) = i > 2
Base.next(a::Instance, i::Int) = (a, i+1)
