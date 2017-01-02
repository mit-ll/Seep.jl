# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
function visit_depth_first(a::ANode, previsit::Function, postvisit::Function, revisit::Function, visited::Set{ANode}=Set{ANode}())
  if sym(a) == :constant return end
  if a ∉ visited
    push!(visited, a)
    previsit(a)
    for b in a.input
      visit_depth_first(b, previsit, postvisit, revisit, visited)
    end
    postvisit(a)
  else
    revisit(a)
  end
end

function find_subgraph(f::ANode, x::ANode...)
  subgraph = Set{ANode}(x)
  path = ANode[]

  add_path() = for i in path push!(subgraph, i) end
  previsit(n::ANode) = push!(path, n)
  revisit(n::ANode) = if n ∈ subgraph add_path() end
  function postvisit(n::ANode)
    @assert pop!(path) == n
    if n ∈ subgraph add_path() end
  end
  visit_depth_first(f, previsit, postvisit, revisit)
  subgraph
end

immutable GradientCache
  objective::ANode
  gradients::Dict{ANode, ANode}
end

gradients(f::ANode) = GradientCache(f, Dict{ANode, ANode}(f=>ones(ANode, size(f))))

constant(a::ANode) = ANode(:constant, (a,), size(a))
do_forward!(a::ANode{:constant}, y, x) = copy!(y, x)

function Base.getindex(c::GradientCache, x::ANode)
  subgraph = find_subgraph(c.objective, x)
  outputs = Dict{ANode, Vector{ANode}}()

  for node in subgraph
    for input in node.input
      if input ∈ subgraph
        push!(get!(outputs, input, ANode[]), node)
      end
    end
  end
  @assert x ∈ subgraph "Invalid gradient.  There is no path between the variable being differentiated and the independent variable.  Technically the gradient is zero, but this usually indicates that you're computing a gradient w.r.t. the wrong node."

  visited = Set{ANode}([c.objective])
  ready = ANode[c.objective]

  while !haskey(c.gradients, x)
    if isempty(ready)
     return zeros(ANode, size(x))
    end
    node = pop!(ready)

    if node ∉ subgraph continue end
    if !haskey(c.gradients, node)
      c.gradients[node] = reduce(+, map(i->gradient_node(i, node, c.gradients[i]), outputs[node]))
    end

    for a in node.input
      if a in visited continue end
      if a ∉ subgraph continue end
      if sym(a) == :constant continue end

      isready = true
      for b in outputs[a]
        if !haskey(c.gradients, b)
          isready = false
          break
        end
      end

      if isready
        push!(ready, a)
        push!(visited, a)
      end
    end
  end
  c.gradients[x]
end
