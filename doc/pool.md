## Memory Pools
Seep allocates memory when a graph is instantiated for internal nodes, result
nodes, input nodes without linked arrays, and input nodes whose linked arrays
are not of the correct type.  By default Seep allocates memory by constructing
new arrays.  To reduce memory usage, and to share memory between instances, you
may pass a memory pool to the instance function.
```julia
x = ANode(4); y = 2*x; z = 2*y;
memory = zeros(Float32, 16)
pool = BuddyPool(memory)

func,dict = instance(pool, z)
```
Seep will use the provided memory to store the data of this instance graph.

After setting the input data to something other than zeros, you can see the
changes reflected in the backing array.
```julia
dict[x][:] = 1:4
println(memory')
```

Seep reclaims memory eagerly within the memory pool.  After running the
instance function, the only arrays that are guaranteed to be populated are
those corresponding to the nodes passed to the instance constructor.
```julia
func()
println(dict[z]') # prints [4.0  8.0  12.0  16.0]
println(dict[y]') # error, y wasn't passed to instance
println(dict[x]') # also prints [4.0  8.0  12.0  16.0]!  Since x was not passed to instance, Seep reused the memory for z
```

Note that you must also be careful not to reference the instance arrays after
the memory pool has been deallocated, since they will still point into the
invalid block.
