# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
immutable TemplateIterator
  type_params::Vector{Symbol}
  types::Vector{Vector{Symbol}}
end

TemplateIterator(t::Vector) =
  TemplateIterator(Symbol[i[1] for i in t], Vector{Symbol}[Symbol[i[2:end]...] for i in t])

Base.start(i::TemplateIterator) = ones(Int, length(i.type_params))
Base.done(i::TemplateIterator, s::Vector{Int}) = s[end] > length(i.types[end])
function Base.next(i::TemplateIterator, s::Vector{Int})
  s = copy(s)
  item = Dict{Symbol,Symbol}([i.type_params[j] => i.types[j][s[j]] for j in 1:length(s)])
  s[1] += 1
  for j in 1:length(s)-1
    if s[j] > length(i.types[j])
      s[j] = 1
      s[j+1] += 1
    end
  end
  item,s
end

hexstring(x::AbstractString) = x[1:6]
hexstring(x::AbstractArray{UInt8}) = join(map(x->hex(x,2), x[1:3]),"")

rsym(ex::Symbol, d::Dict{Symbol,Symbol}) = get(d, ex, ex)
rsym(ex::Expr, d::Dict{Symbol,Symbol}) = Expr(ex.head, map(x->rsym(x, d), ex.args)...)
rsym(t::Tuple, d::Dict{Symbol,Symbol}) = rsym(t[1], d), t[2]

type_names(templates, d::Dict{Symbol,Symbol}) = join([d[i[1]] for i in templates], "_")
replace_type_parameters(signature, d::Dict{Symbol,Symbol}) = map(t->rsym(t, d), signature)

function old_templates(templates::Expr)
  @assert templates.head == :vect
  for i in 1:length(templates.args)
    @assert templates.args[i].head == :tuple
  end
  [(a.args...) for a in templates.args]
end

function curly_templates(x::Expr)
  @assert x.head == :curly
  templates = sizehint!(Tuple[], length(x.args)-1)
  for i in 2:length(x.args)
    @assert isa(x.args[i], Expr) && x.args[i].head == :<:
    @assert isa(x.args[i].args[1], Symbol)

    if x.args[i].args[2] == :Real
      push!(templates, (x.args[i].args[1], :Cfloat, :Cdouble))
    elseif isa(x.args[i].args[2], Expr) && x.args[i].args[2].head == :curly && x.args[i].args[2].args[1] == :Union
      push!(templates, (x.args[i].args[1], x.args[i].args[2].args[2:end]...))
    else
      error("Invalid cuda_kernel template argument: $(x.args[i])")
    end
  end
  return templates
end

macro cuda_kernel(funcall, va...)
  @assert funcall.head == :call
  for i in 2:length(funcall.args)
    @assert funcall.args[i].head == :(::)
  end
  signature = map(x->(x.args[2], x.args[1]), funcall.args[2:end])

  if isa(funcall.args[1], Expr) && funcall.args[1].head == :curly
    @assert length(va) == 1
    @assert isa(funcall.args[1].args[1], Symbol)
    name = string(funcall.args[1].args[1])
    templates = curly_templates(funcall.args[1])
    text = va[1]
  elseif isa(funcall.args[1], Symbol)
    @assert length(va) == 2
    name = string(funcall.args[1])
    templates = old_templates(va[1])
    text = va[2]
  else
    error("Invalid cuda template")
  end

  cuda_kernel(name, signature, templates, eval(text))
end

macro cuda_gsl(funcall, text)
  quote
    @cuda_kernel($funcall, """
  int i;
  for (i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
    $($(esc(text)));
  }""")
  end
end
