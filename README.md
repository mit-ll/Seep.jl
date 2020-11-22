# Seep.jl
Because not everybody's data is big.

## Basics: ANodes, Arrays, and Instances
Seep builds and evaluates computational flow graphs in julia.  A computational
flow graph (CFG) is a directed acyclic graph where nodes represent values and
edges represent data dependencies.  It is similar to the Single Static
Assignment form of a program used by compilers, but there is no control flow.
All nodes are evaluated each time the graph is executed.

The CFG is first defined as an *abstract graph* using `ANodes`.  The *abstract
graph* specifies the size and shape of each variable, how they are connected,
and how each value will be computed.  It does not allocate storage or define
the order of operations.

One or more  *instances* of may be constructed from the *abstract graph*.  Each
*instance* is a julia function that evaluates the CFG described by the
*abstract graph*.  Storage for the values of each `ANode` is allocated
statically when the *instance* is constructed.  In addition to being callable
as a function, the *instance* provides access to the Arrays on which it
operates.

### Creating The Abstract Graph
*Abstract graphs* are built from `ANode`s.  The first `ANode` defined is always
an input `ANode` (the constructors of all of the other types require `ANode`s
as arguments).  Input `ANode`s are build by calling the `ANode(name::String,
dims::Int...)` constructor.  The name is optional, but it helps to make any
code you will write using the instance a bit cleaner and sometimes also faster.

To get started, let's create an `ANode` to hold a single element, and call it `x`.
```julia
julia> x = ANode("x", 1)
```

Since it's often useful to create a `ANode` and assign it to a variable of the
same name, there is a macro called `@named` to do exactly that.  Let's create
another `ANode` named `y` using the `@named` macro.
```julia
julia> @named y = ANode(1)
```

`ANode`s can be treated as if they were arrays in many cases.  You can't index
them (i.e. `x[1]` won't work) since they're abstract, but you can perform
arithmetic on them.  The result of operating on `ANode`s is always a new
`ANode` that represents the computed value.

Let's create a couple more `ANode`s with compute simple functions of `x` and `y`.
```julia
julia> @named begin
  a = x + y
  b = 2a - log(y)
end
```

This creates 4 new `ANode`s: `a`, `2a`, `log(y)`, and `b`.  Only `a` and `b`
are named and assigned to variables in the workspace.  `2a` and `log(y)` are
referenced by `b`, but not otherwise visible.

### Instantiating the Graph
So far, we've created six `ANode`s.  Let's instantiate the graph so we can use them.

```julia
julia> graph = instance(a, b)
```

Each `ANode` passed to the instance constructor will be evaluated exactly once
when the instance is evaluated.  All of the nodes they reference (e.g. `2a` and
`log(y)`) will also be evaluated as necessary.

### Evaluating the Graph
To use the graph, we first have to provide values for the input nodes.  The arrays
that were allocated when the *instance* was created are available as fields of the
*instance*.  Let's populate the input `ANode`s' backing arrays with some data.

```julia
julia> graph.x[1] = 1
julia> graph.y[1] = 2
```

Now that the inputs are populated, we can evaluate the *instance* by calling it
as a function.
```julia
julia> graph()
```

Finally, we can use the results by inspecting the arrays where the results are stored.
```julia
julia> println(graph.a)
julia> println(graph.b)
```

## More Information
This has been a short introduction that shows only the most basic features of
Seep.  For mor information, see the [doc/](doc/) directory in the root of the
source tree.
 - [Supported Operations on `ANodes`](doc/operators.md)
 - [Automatic Differentiation (and Gradient Descent)](doc/autodiff.md)
 - [Memory Pools](doc/pool.md)
 - [Convolution and Pooling](doc/conv2.md)
 
## Disclaimer.   
DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.   

© 2017 MASSACHUSETTS INSTITUTE OF TECHNOLOGY.   
* Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).   
* SPDX-License-Identifier: MIT.   
 
This material is based upon work supported by the Undersecretary of Defense for Research and Engineering under Air Force Contract No. FA8721-05-C-0002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of USD(R&E).    

The software/firmware is provided to you on an As-Is basis.   
