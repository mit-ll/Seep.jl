# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
type ANode{S,A}
  id::Int
  name::Compat.UTF8String
  input::Vector{ANode}
  size::Tuple{Vararg{Int}}

  arg::A

  control::Vector{ANode}

  @compat (::Type{ANode}){T}(t::Type{T}, x::T; c::Bool=false) = new{c ? :const : :load, T}(0, "", [], size(x), x, [])
  @compat (::Type{ANode}){T}(t::Type{T}, x; c::Bool=false) = new{c ? :const : :load, T}(0, "", [], size(x), T(x), [])
  @compat (::Type{ANode})(x; c::Bool=false) = ANode(typeof(x), x; c=c)

  @compat (::Type{ANode})(x::Int...) = ANode(x)
  @compat (::Type{ANode})(x::Tuple{Vararg{Int}}) = new{:input,Void}(0, "", [], x, nothing, [])

  @compat (::Type{ANode})(s::Symbol, input::Tuple{Vararg{ANode}}, size::Tuple{Vararg{Int}}, arg=nothing) =
	new{s,typeof(arg)}(0, "", collect(input), size, arg, [])
end

@compat (::Type{ANode})(name::AbstractString, x...) = name!(ANode(x...), name)

sym{S}(::ANode{S}) = S
nin(n::ANode) = length(n.input)

id(n::ANode) = n.id
name(n::ANode) = n.name
input(n::ANode) = n.input
arg(n::ANode) = n.arg
mutates(n::ANode) = false
name!(n::ANode, s::AbstractString) = (n.name = utf8(s); n)

function named_(e::Expr)
  if e.head in (:block, :const, :global)
    return Expr(e.head, map(named_, e.args)...)
  elseif e.head == :(=) && isa(e.args[1], Symbol)
    return :($(esc(e.args[1])) = name!($(esc(e.args[2])), $(string(e.args[1]))))
  elseif e.head == :line
    return e
  else
    error("unexpected expression: $e")
  end
end

macro named(arg...)
  if length(arg) == 2 && (isa(arg[1], AbstractString) || isa(arg[1], Symbol))
    name,node = arg
    return :($(esc(Symbol(name))) = name!($(esc(node)), $(string(name))))
  elseif length(arg) == 1 && isa(arg[1], Expr)
    return named_(arg[1])
  else
    error("unexpected arguments to Seep.@named")
  end
end

function order{S,A}(n::AbstractArray{ANode{S,A}})
  for i in 2:length(n)
    push!(n[i].control, n[i-1])
  end
  n[end]
end

function order(n::ANode...)
  for i in 2:length(n)
    push!(n[i].control, n[i-1])
  end
  n[end]
end

function noop(nodes::AbstractVector{ANode})
  n = ANode(:noop, (), ())
  append!(n.control, nodes)
  n
end

function noop(nodes::ANode...)
  n = ANode(:noop, (), ())
  append!(n.control, collect(ANode, nodes))
  n
end

Base.size(x::ANode) = x.size
Base.size(x::ANode, i::Integer) = i <=  length(x.size) ? x.size[i] : 1
Base.length(x::ANode) = prod(x.size)
Base.ndims(x::ANode) = length(x.size)

gradient_node(a::ANode, wrt::Union{ANode,Int}, b::ANode) = b.*gradient_node(a, wrt)

function gradient_node(a::ANode, wrt::ANode, b::ANode)
  for i in 1:length(a.input)
    if wrt == a.input[i]
      return gradient_node(a, Val{i}, b)
    end
  end
  error("Invalid gradient node")
end

connected_nodes(n::ANode...) = connected_nodes(collect(ANode, n))
connected_nodes(n::ANode, visited=Set{ANode}()) = connected_nodes(ANode[n], visited)

function remove_visited_nodes(q::Vector{ANode}, visited::Set{ANode})
  N = length(q)
  i = 1
  while i <= N && !(q[i] in visited)
    i += 1
  end

  if i <= N
    j = i
    for i in j+1:N
      if !(q[i] in visited)
        q[j] = q[i]
        j += 1
      end
    end

    resize!(q, j-1)
  end
end

function connected_nodes{A<:ANode}(n::Vector{A}, visited=Set{ANode}())
  q = ANode[n;]
  while !isempty(q)
    nn = pop!(q)
    if nn in visited
      remove_visited_nodes(q, visited)
      continue
    end
    push!(visited, nn)
    append!(q, nn.input)
    append!(q, nn.control)
  end

  return visited
end

function output_map(nodes)
  inputs = Dict{ANode,Vector{ANode}}(map(x->x=>ANode[], nodes))
  outputs = Dict{ANode,Vector{ANode}}(map(x->x=>ANode[], nodes))

  for a in nodes
    inputs[a] = unique(filter(x->haskey(inputs, x), [a.input; a.control]))
    for b in inputs[a]
      push!(outputs[b], a)
    end
  end

  for mutator in filter(mutates, nodes)
    a = mutator.input[1]
    if !haskey(outputs, a)
      # a is not a part of this subgraph.
      continue
    end
    out = outputs[a]
    @assert count(mutates, out) == 1 "Too many mutators attached to $a"
    # a has a mutator.  all of its other output must be executed before the mutator.
    for b in out
      if b == mutator continue end
      if mutator in outputs[b] continue end
      push!(inputs[mutator], b)
      push!(outputs[b], mutator)
    end
  end

  return outputs,inputs
end

toposort(n::ANode...) = toposort(collect(ANode, n))

function toposort{A<:ANode}(n::Vector{A}, limit=Union{}[])
  unsorted = connected_nodes(n, Set{ANode}(limit))
  outputs,inputs = output_map(unsorted)

  ready = Set{ANode}(keys(filter((node,input)->isempty(input), inputs)))
  union!(ready, limit)

  sorted = sizehint!(Array{ANode}(0), length(unsorted))
  while !isempty(ready)
    a = pop!(ready)

    if sym(a) != :noop && a∉limit
      push!(sorted, a)
    end

    delete!(unsorted, a)

    for b in outputs[a]
      if b in limit continue end
      @assert !(b in ready)
      if !any(x->x in unsorted, inputs[b])
        push!(ready, b)
      end
    end
  end

  @assert isempty(unsorted)

  return sorted
end

function connected_components(nodes::Vector{ANode},
  outputs::Dict{ANode,Vector{ANode}}, inputs::Dict{ANode,Vector{ANode}})
  cc = Vector{ANode}[]
  s = Set{ANode}(nodes)
  while !isempty(s)
    n = first(s)
    q = ANode[n]
    cc1 = ANode[]
    while !isempty(q)
      i = pop!(q)
      if i in s
        delete!(s, i)
        push!(cc1, i)
        append!(q, inputs[i])
        append!(q, outputs[i])
      end
    end
    push!(cc, cc1)
  end
  return cc
end

function Base.show(io::IO, n::ANode)
  if !isempty(n.name)
    print(io, "ANode{$(sym(n))}($(n.id), $(n.name))")
  else
    print(io, "ANode{$(sym(n))}($(n.id))")
  end
end

graphviz(fn::AbstractString, nn...) = open(f->graphviz(f, nn...), fn, "w")
graphviz(io::IO, nn::ANode...) = graphviz(io, collect(ANode, nn))

function graphviz{A<:ANode}(io::IO, nn::Vector{A})
  println(io, "digraph {")

  unsorted = connected_nodes(nn)

  inputs = Dict{ANode,Vector{ANode}}()
  outputs = Dict{ANode,Vector{ANode}}()
  special = Dict{ANode,Vector{ANode}}()

  id = 1
  for a in unsorted
    a.id = id
    id += 1
    inputs[a] = collect(a.input)
    append!(inputs[a], a.control)

    for b in inputs[a]
      push!(get!(outputs, b, ANode[]), a)
    end
  end

  for a in unsorted
    if mutates(a)
      @assert count(mutates, outputs[a.input[1]]) == 1 "Too many mutators attached to $(a.input[1])"
      ca = connected_nodes(a)

      for b in outputs[a.input[1]]
        if b != a && !(b in ca)
          push!(get!(special, a, ANode[]), b)
        end
      end
    end
  end

  for n in unsorted
    label = (isempty(n.name) ? "" : string(n.name, "\\n"))*"$(sym(n)) ($(n.id))"
    color = elementwise(n) ? "black" : "red"
    println(io, "  n$(n.id) [label=\"$label\",color=\"$color\"];")

    for i in 1:length(n.input)
      println(io, "    n$(n.input[i].id) -> n$(n.id);")
    end

    for i in 1:length(n.control)
      println(io, "    n$(n.control[i].id) -> n$(n.id) [style=dashed];")
    end

    if haskey(special, n)
      for i in 1:length(special[n])
        println(io, "    n$(special[n][i].id) -> n$(n.id) [style=dotted];")
      end
    end
  end

  println(io, "}")
end
