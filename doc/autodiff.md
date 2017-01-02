## Automatic Differentiation
Seep can automatically compute the gradient of any node with respect to any
other node.  This computation is done on the abstract graph.  The gradients
function returns a map of gradients of one variable.
```julia
gradient_map = gradients(z)
```
The map can be queried as if it were a dictionary.
```julia
dzdx = gradient_map[x]
dzdy = gradient_map[y]
```
The returned values are ANodes and can be used like ANodes that you create with
the overloaded operators.  To evaluate the gradients, create an instance that
includes the gradient nodes.
```julia
func,dict = instance(z, dzdx, dzdy)
func()
println(dict[dzdx], dict[dzdy])
```
