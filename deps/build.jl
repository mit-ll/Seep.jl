#!/usr/bin/env julia
# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.

if isempty(ARGS)
  if Pkg.installed("CUDArt") == nothing
    exit(0)
  end
  const file,io = mktemp()
else
  const file = ARGS[1]
  const io = open(file, "w")
end

using Compat, CUDArt, JLD, SHA

include(Pkg.dir("Seep", "src", "cudahacks_common.jl"))

macro cuda(x) end
macro llvm(x) end

macro cuda_text(text)
   quote
       println(io, $text)
   end
end

macro cuda_include(filename)
   quote
       println(io, "#include \"$($filename)\"")
   end
end

function ctype(ex::Symbol)
    if ex == :Void return "void" end
    if string(ex)[1] == 'C'
        return Symbol(string(ex)[2:end])
    end
    string(ex)
end

function ctype(ex::Expr)
    if (ex.head == :&)
        return string(ctype(ex.args[1]), "* __restrict__")
    end
    error("??")
end

function cuda_kernel(name, signature, templates, text)
    if (!isempty(templates))
        template_args = templates |> x->map(x->"typename $(first(x))", x) |> x->join(x, ", ")
        println(io, "template <$template_args>")
    end

    hashname = "$(name)_$(hexstring(sha1(text)))"
    fullsig = signature |> x->map(x->"$(ctype(x[1])) $(x[2])", x) |> x->join(x, ", ")
    argnames = signature |> x->map(x->string(x[2]), x) |> x->join(x, ", ")

    println(io, "__device__ void $hashname($fullsig) {")
    println(io, text)
    println(io, "}")
    println(io)

    if (!isempty(templates))
        println(io, "extern \"C\" {")
        for type_map in TemplateIterator(templates)
            typed_sig = replace_type_parameters(signature, type_map)
            fullsig = typed_sig |> x->map(x->"$(ctype(x[1])) $(x[2])", x) |> x->join(x, ", ")

            print(io, "__global__ void $(hashname)_$(type_names(templates, type_map))($fullsig) { ")
            print(io, "$hashname($argnames);")
            println(io, " }")
        end
        println(io, "}")
        println(io)
    end
end

include(Pkg.dir("Seep", "src", "sources.jl"))

close(io)
compiler = haskey(ENV, "CXX") ? [string("-ccbin=", ENV["CXX"])] : []
run(`nvcc -ptx -O3 --use_fast_math -o $(Pkg.dir("Seep", "kernels.ptx")) $compiler -x cu -I$(Pkg.dir("Seep", "src")) $file`)

if isempty(ARGS)
    rm(file)
end
