# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
for a in ("", "t", "c")
  for b in ("", "t", "c")
    local sym = a == b == "" ? :* : Symbol("A", a, "_mul_B", b)
    if !isdefined(Base, sym) continue end
    q = QuoteNode(sym)

    @eval begin
      function Base.$(sym)(x::ANode, y::ANode)
        @assert size(x, $(a == "" ? 2 : 1)) == size(y, $(b == "" ? 1 : 2))
        ANode($q, (x, y), (size(x, $(a == "" ? 1 : 2)), size(y, $(b == "" ? 2 : 1))))
      end

      function do_forward!(n::ANode{$q}, out, in1, in2)
        BLAS.gemm!($(a == "" ? 'N' : 'T'), $(b == "" ? 'N' : 'T'), one(eltype(out)), in1, in2, zero(eltype(out)), out)
      end

      gradient_node(n::ANode{$q}, ::Type{Val{1}}, bp::ANode) = $(a == "" ? (b=="" ? :(bp*n.input[2].') : :(bp*n.input[2])) : (b=="" ? :(n.input[2]*bp.') : :(n.input[2].'*bp.')))

      gradient_node(n::ANode{$q}, ::Type{Val{2}}, bp::ANode) = $(b == "" ? (a=="" ? :(n.input[1].'*bp) : :(n.input[1]*bp)) : (a=="" ? :(bp.'*n.input[1]) : :(bp.'*n.input[1].')))
    end
  end
end
