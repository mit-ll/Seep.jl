# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
Base.zeros(::Type{ANode}, sz::Int...) = ANode(:zeros, (), sz)
Base.zeros(::Type{ANode}, sz::Tuple{Vararg{Int}}) = ANode(:zeros, (), sz)
do_forward!(::ANode{:zeros}, out) = fill!(out, zero(eltype(out)))
elementwise(::ANode{:zeros}) = true
@llvm llvm_val(n::ANode{:zeros}, ::ModuleRef, b::BuilderRef, t::Type) = llvm_const(t, 0)

Base.ones(::Type{ANode}, sz::Int...) = ANode(:ones, (), sz)
Base.ones(::Type{ANode}, sz::Tuple{Vararg{Int}}) = ANode(:ones, (), sz)
do_forward!(::ANode{:ones}, out) = fill!(out, one(eltype(out)))
elementwise(::ANode{:ones}) = true
@llvm llvm_val(n::ANode{:ones}, ::ModuleRef, b::BuilderRef, t::Type) = llvm_const(t, 1)

Base.fill!(a::ANode, value) = ANode(:fill!, (a,), size(a), convert(eltype(a), value))
mutates(::ANode{:fill!}) = true
do_forward!(a::ANode{:fill!}, out) = fill!(out, a.arg)
gradient_node!(a::ANode{:fill!}, wrt::ANode, b) = zeros(size(wrt))
#elementwise(::ANode{:fill!}) = true
#@llvm llvm_val(n::ANode{:fill!}, ::ModuleRef, b::BuilderRef, t::Type) = llvm_const(t, n.arg)

Base.rand!(a::ANode) = ANode(:rand!, (a,), size(a))
mutates(::ANode{:rand!}) = true
do_forward!(a::ANode{:rand!}, out) = rand!(out)
gradient_node!(a::ANode{:rand!}, wrt::ANode, b) = zeros(size(wrt))

Base.randn!(a::ANode) = ANode(:randn!, (a,), size(a))
mutates(::ANode{:randn!}) = true
do_forward!(a::ANode{:randn!}, out) = randn!(out)
gradient_node!(a::ANode{:randn!}, wrt::ANode, b) = zeros(size(wrt))
