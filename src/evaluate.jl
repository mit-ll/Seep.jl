# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
immutable EvaluatedInstance{T} <: Instance
  storage::Dict{ANode,T}
end

evaluate(n::ANode...) = evaluate(collect(ANode, n))
evaluate(T::Type, n::ANode...) = evaluate(T, collect(ANode, n))
evaluate(pool::StoragePool, n::ANode...) = evaluate(pool, collect(ANode, n))

evaluate{A<:ANode}(n::Vector{A}) = evaluate(Array{Float64}, n)
evaluate{A<:ANode}(T::Type, n::Vector{A}) = evaluate(NullPool(T), n)

function evaluate{A<:ANode}(pool::StoragePool, n::Vector{A})
  sorted = toposort(n)

  T = atype(pool)
  dict = Dict{ANode,T}()

  refcnt = Dict{ANode,Vector{Int}}()
  for a in n
    if sym(a) == :noop continue end
    get!(refcnt, output_node(a), Int[0])[1] += 1
  end

  for a in sorted
    for b in a.input
      get!(refcnt, output_node(b), Int[0])[1] += 1
    end
  end

  for i in sorted
    @assert refcnt[output_node(i)][1] > 0

    if sym(i) == :input
      error("Input nodes are not allowed here (try passing an array to the ANode constructor).")
    elseif sym(i) == :const || sym(i) == :load
      if isa(arg(i), T)
        dict[i] = arg(i)
      else
        get!(refcnt, i, Int[0])[1] += 1
        t = dict[i] = allocate(pool, i.size...)
        copy!(t, arg(i))
      end
    else
      if mutates(i)
        dict[i] = dict[i.input[1]]
        do_forward!(i, map(j->dict[j], i.input)...)
      else
        dict[i] = allocate(pool, i.size...)
        do_forward!(i, dict[i], map(j->dict[j], i.input)...)
      end
    end

    for a in i.input
      aa = output_node(a)
      @assert refcnt[aa][1] > 0

      refcnt[aa][1] -= 1
      if refcnt[aa][1] == 0 && arg(aa) != dict[aa]
        deallocate(pool, dict[aa])
        delete!(dict, aa)
      end
    end
  end

  for (n,r) in refcnt
    if r[1] == 0
      continue
    elseif r[1] == 1
      r[1] -= 1
      if dict[n] !== arg(n)
        deallocate(pool, dict[n])
      end
    else
      error("invalid refcnt")
    end
  end

  return EvaluatedInstance(dict)
end
