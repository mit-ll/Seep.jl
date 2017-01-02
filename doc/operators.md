## Supported Operators
The following operators are supported:
  - Element-wise binary operators: `+`, `-`, `.+`, `.-`, `.*`, `./`, and `.^` (like julia, the dot versions broadcast if necessary)
  - Matrix multiplication: `*` (and `At_mul_B`, `A_mul_Bt`, and `At_mul_Bt`)
  - Activation functions: `tanh`, `sigm`, `log`, `exp`, `sqrt`
  - Reductions: `sum`, `minimum`, and `maximumum` over the whole array (`sum(::ANode)`) or over one dimension (`sum(::ANode, dim::Int)`)
  - CNN: `conv2(filter, data; stride)` and `pool(data, size; stride)`, `max!(::ANode, dim::Int`, `min!(::ANode, dim::Int)`
  - Indexing: `getindex(::ANode, i::Array{Int})`
  - Comparison: `.==` and `.!=` (comparing to ANodes or an ANode and a Real)
  - Other functions: `softmax(x)` and `softmax(x, dim::Int)`

Many more complicated functions can be build from these primitives.  Sigmoid
and softmax functions, for example, are simple julia functions that operate on
numbers, arrays, or ANodes.
```julia
sigm(x) = 1./(1+exp(-x))
softmax(x) = let z = exp(x); z./sum(z) end
softmax(x,dim) = let z = exp(x); z./sum(z, dim) end
```
