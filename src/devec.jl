# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
elementwise(::ANode) = false

for i in [:.+, :.-, :.*, :./, :.^]
  @eval elementwise(::ANode{$(QuoteNode(i))}) = true
end

for i in [:+, :*]
  @eval elementwise(::ANode{$(QuoteNode(symbol("c", i)))}) = true
end

for i in [:/, :^]
  @eval elementwise(::ANode{$(QuoteNode(symbol("c", i)))}) = true
  @eval elementwise(::ANode{$(QuoteNode(symbol(i, "c")))}) = true
end

# Create a devectorized kernel that evaluates node n
# taking the nodes [inp] as input.  Returns a Symbol.
function devec_kernel(n::ANode, inp::Vector{ANode})
  b = :(for i in 1:length(out) end)

  symtab = Dict{ANode,Any}()
  symtab[n] = :(out[i])
  for i in 1:length(inp)
    symtab[inp[i]] = :(input[$i][i])
  end

  ready = ANode[n]
  done = Set{ANode}([ready;  inp])
  while !isempty(ready)
    i = pop!(ready)
    out = get!(gensym, symtab, i)
    arg = map(x->get!(gensym, symtab, x), i.input)

    if sym(i) == Symbol("c*")
      unshift!(b.args[2].args, :($out = *($(i.arg), $(arg...))))
    elseif sym(i) == Symbol("c+")
      unshift!(b.args[2].args, :($out = +($(i.arg), $(arg...))))
    elseif sym(i) == Symbol("c^")
      unshift!(b.args[2].args, :($out = ^($(i.arg), $(arg...))))
    elseif sym(i) == Symbol("^c")
      unshift!(b.args[2].args, :($out = ^($(arg...), $(i.arg))))
    else
      unshift!(b.args[2].args, :($out = $(sym(i))($(arg...))))
    end

    for j in i.input
      if !(j in done)
        push!(done, j)
        push!(ready, j)
      end
    end
  end

  #@show b

  s = gensym("devec")
  @eval @inline do_forward!(n::ANode{$(QuoteNode(s))}, out, input...) = @fastmath @inbounds $b
  return s
end

function add_color(colors::Dict{ANode,Set{ANode}}, node::ANode, c::ANode)
  set = get!(Set{ANode}, colors, node)
  if c in set return end
  push!(set, c)
  for i in node.input add_color(colors, i, c) end
  for i in node.control add_color(colors, i, c) end
  return
end

function intern_values{K,V}(d::Dict{K,V})
  cache = Dict{V,V}()
  for (k,v) in d
    if haskey(cache, v)
      d[k] = cache[v]
    else
      cache[v] = v
    end
  end
  return d
end

devectorize(n::ANode...) = devectorize(collect(ANode, n))
function devectorize(n::Vector{ANode})
  nodes = connected_nodes(n)
  outputs = Dict{ANode,Vector{ANode}}()
  for i in nodes
    get!(Vector{ANode}, outputs, i)
    for j in i.input push!(get!(Vector{ANode}, outputs, j), i) end
    for j in i.control push!(get!(Vector{ANode}, outputs, j), i) end
  end

  colors = Dict{ANode,Set{ANode}}()
  for i in n
    add_color(colors, i, i)
  end
  intern_values(colors)

  inelligible = Set{ANode}(n)
  for (i,output) in outputs
    c = colors[i]
    if !elementwise(i)
      #println("!elementwise $i")
      push!(inelligible, i)
    elseif !all(elementwise, output)
      #println("!elementwise output $i")
      push!(inelligible, i)
    elseif any(j->colors[j]!==c, output)
      #println("multicolor output $i")
      push!(inelligible, i)
    end
  end

  replace = Dict{ANode,ANode}()

  for i in collect(inelligible)
    s = copy(inelligible)
    delete!(s, i)
    a = setdiff(connected_nodes(i, s), inelligible)
    inp = Set{ANode}(i.input)
    for j in a
      for k in j.input
        push!(inp, k)
      end
    end
    setdiff!(inp, a)

    if !isempty(a)
      #@show i
      #@show a
      #@show inp

      d = ANode(devec_kernel(i, collect(inp)), (inp...), size(i))
      for j in outputs[i]
        for m in 1:length(j.input)
          if j.input[m] === i
            j.input[m] = d
          end
        end

        for m in 1:length(j.control)
          if j.control[m] === i
            j.control[m] = d
          end
        end
      end

      delete!(inelligible, i)
      push!(inelligible, d)
      replace[i] = d
      #println()
    end
  end

  return map(x->get(replace, x, x), n)
end
