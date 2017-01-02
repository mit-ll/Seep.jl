# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
if filemode(Pkg.dir("Seep", "kernels.ptx")) == 0
  error("Seep GPU (kernels.ptx) can not be found. Try Pkg.build(\"Seep\")")
end

const md = CuModuleFile(Pkg.dir("Seep", "kernels.ptx"))
#__init__() = global md = CuModule(Pkg.dir("Seep") * "/kernels.ptx", false)

cutype(ex::Symbol) = ex
cutype(ex::Expr) = ex.head == :(&) ? :(CudaArray{$(ex.args[1])}) : ex

@compat (::Type{CudaArray{T}}){T}(i::Integer...) = CudaArray(T, i...)
Base.ones(a::CudaArray) = let s = similar(a); fill!(s, one(eltype(s))); s end
Base.zeros(a::CudaArray) = let s = similar(a); fill!(s, zero(eltype(s))); s end
@compat (::Type{CudaArray{T}}){T}(::Type{CudaArray{T}}, i::Int...) = CudaArray(T, i...)
Base.eltype{T}(::Type{CudaArray{T}}) = T

Base.reshape(x::CudaArray, i...) = reinterpret(eltype(x), x, i)

include("cudahacks_common.jl")

const STREAM=CuStream[CUDAdrv.default_stream()]

function cuda_kernel(name, signature, templates, text)
    block = quote
        global $(Symbol(name))
    end

    hashname = "$(name)_$(hexstring(sha1(text)))"

    if isempty(templates)
        push!(block.args, "no templates")
    else
        for type_map in TemplateIterator(templates)
            typed_sig = replace_type_parameters(signature, type_map)
            kernel_name = "$(hashname)_$(type_names(templates, type_map))"

            push!(block.args, :(try
                    global const $(Symbol(kernel_name)) = CuFunction(md, $kernel_name)
                catch e
                    println($("Unable to load cuda function for $kernel_name.  Try Pkg.build(\"Seep\")."))
                    rethrow(e)
                end
            ))

            ca = :($(Symbol(name))())
            for i in typed_sig
                push!(ca.args, :($(i[2])::$(cutype(i[1]))))
            end

            def = :(CUDAdrv.launch($(Symbol(kernel_name)), CUDAdrv.CuDim3(Int(div(n+127,128))), CUDAdrv.CuDim3(128), (), stream=STREAM[1]))
            for a in typed_sig
                if eval(cutype(a[1])) <: CudaArray
                   push!(def.args[end-1].args, :(Base.unsafe_convert(Ptr, $(a[2]).ptr)))
                else
                   push!(def.args[end-1].args, a[2])
                end
            end

            push!(block.args, :($ca = $def))
        end
    end

    return block
end
