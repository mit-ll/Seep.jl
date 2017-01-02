global const check_syms = Tuple{Symbol,Symbol}[]

isexpr(x::Expr) = true
isexpr(x::Any) = false
isexpr(x::Expr, s::Symbol) = x.head == s
isexpr(x::Any, s::Symbol) = false

istype(x::Expr) = x.head == :curly
istype(::Symbol) = true

macro fun(expr)
  @assert isexpr(expr)
  if expr.head == :(=)
    lhs,rhs = expr.args
    if isexpr(lhs, :(:))
      error("found : where :: was expected")
    end
    @assert isexpr(lhs, :(::))
    call,rettype = lhs.args
    @assert istype(rettype)
    @assert isexpr(rhs, :tuple)
    rhargs = rhs.args
  elseif expr.head == :(::)
    call,rettype = expr.args
    @assert istype(rettype)
    rhargs = call.args[2:end]
  else
    error("Not a @fun expression ($(expr.head))")
  end

  @assert isexpr(call, :call)
  fname = call.args[1]
  jargs = call.args[2:end]
  jdict = Dict{Symbol,Expr}()
  for i in jargs
    ta = isexpr(i, :kw) ? i.args[1] : i
    if isexpr(ta, :(:))
      error("found : where :: was expected")
    end
    @assert isexpr(ta, :(::))
    @assert isa(ta.args[1], Symbol)
    jdict[ta.args[1]] = ta
  end

  a = quote
    #export $(esc(fname))
    $(esc(call)) = ccall(Libdl.dlsym($lib, $(string(fun_prefix, fname))), $rettype, ())
  end

  b = a.args[2]
  @assert isexpr(b, :(=))
  cc = b.args[2]
  @assert isexpr(cc, :ccall)

  ccall_params = cc.args
  @assert isexpr(ccall_params[3], :tuple)
  ccall_types = ccall_params[3].args
  for i in rhargs
    ta2 = isa(i, Symbol) ? jdict[i] : i
    if isexpr(ta2, :(:))
      error("found : where :: was expected")
    end
    @assert isexpr(ta2, :(::))
    push!(ccall_types, esc(ta2.args[2]))
    push!(ccall_params, esc(ta2.args[1]))
  end

  push!(check_syms, (lib, symbol(fun_prefix, fname)))

  return a
end
