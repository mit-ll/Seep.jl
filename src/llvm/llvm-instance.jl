# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
llvm_const{T<:Signed}(::Type{T}, r::Integer) = LLVM.ConstInt(TypeRef(T), r, true)
llvm_const{T<:Unsigned}(::Type{T}, r::Integer) = LLVM.ConstInt(TypeRef(T), r, false)
llvm_const{T<:Real}(::Type{T}, r::Real) = LLVM.ConstReal(TypeRef(T), r)

function intrinsic(mod::ModuleRef, name::AbstractString, rt::Type, params::Type...)
  fn = LLVM.GetNamedFunction(mod, name)
  if fn.ptr != C_NULL return fn end
  fn_type = LLVM.FunctionType(TypeRef(rt), collect(TypeRef, params))
  return LLVM.AddFunction(mod, name, fn_type)
end

function build_grid_stride_loop(mod::ModuleRef, name::AbstractString, eltype::Type, outs::Vector{ANode}, ins::Vector{ANode})
  pt = LLVM.PointerType(LLVM.TypeRef(eltype), 0)
  fun = LLVM.AddFunction(mod, name, LLVM.FunctionType(TypeRef(Void), [TypeRef(Int32); fill(pt, length(outs)+length(ins))]))
  LLVM.AddNamedMetadataOperand(mod, "nvvm.annotations",
    LLVM.MDNode(ValueRef[fun, LLVM.MDString("kernel"), llvm_const(UInt32, 1)]))

  sorted = Seep.toposort(outs, ins)

  limit = LLVM.GetFirstParam(fun)
  LLVM.SetValueName(limit, "limit")

  inptrs = Dict{ANode,ValueRef}()
  outptrs = Dict{ANode,ValueRef}()

  param = LLVM.GetNextParam(limit)
  for i in outs
    outptrs[i] = param
    LLVM.SetValueName(param, string(i.name, "_out"))
    param = LLVM.GetNextParam(param)
  end

  for i in ins
    inptrs[i] = param
    LLVM.SetValueName(param, string(i.name, "_in"))
    param = LLVM.GetNextParam(param)
  end

  builder = LLVM.CreateBuilder()
  entry_block = LLVM.AppendBasicBlock(fun, "entry")
  test_block = LLVM.AppendBasicBlock(fun, "test")
  loop_block = LLVM.AppendBasicBlock(fun, "loop")
  exit_block = LLVM.AppendBasicBlock(fun, "exit")

  # entry:
  LLVM.PositionBuilderAtEnd(builder, entry_block)
  tid = LLVM.BuildCall(builder, intrinsic(mod, "llvm.nvvm.read.ptx.sreg.tid.x", Int32), ValueRef[], "tid")
  ntid = LLVM.BuildCall(builder, intrinsic(mod, "llvm.nvvm.read.ptx.sreg.ntid.x", Int32), ValueRef[], "ntid")
  ctaid = LLVM.BuildCall(builder, intrinsic(mod, "llvm.nvvm.read.ptx.sreg.ctaid.x", Int32), ValueRef[], "ctaid")
  nctaid = LLVM.BuildCall(builder, intrinsic(mod, "llvm.nvvm.read.ptx.sreg.nctaid.x", Int32), ValueRef[], "nctaid")
  first_index = LLVM.BuildAdd(builder, LLVM.BuildMul(builder, ctaid, ntid, ""), tid, "first_index")
  stride = LLVM.BuildMul(builder, ntid, nctaid, "stride")
  LLVM.BuildBr(builder, test_block)

  # test:
  LLVM.PositionBuilderAtEnd(builder, test_block)
  index = LLVM.BuildPhi(builder, TypeRef(Int32), "index")
  LLVM.AddIncoming(index, [first_index], [entry_block])
  c_if = LLVM.BuildICmp(builder, LLVM.IntULE, index, limit, "continue_if")
  LLVM.BuildCondBr(builder, c_if, loop_block, exit_block)

  # loop:
  LLVM.PositionBuilderAtEnd(builder, loop_block)
  dict = Dict{ANode,ValueRef}()
  for (a,ptr) in inptrs
    addr = LLVM.BuildInBoundsGEP(builder, ptr, [index], string(a.name, "_in[index]"))
    dict[a] = LLVM.BuildLoad(builder, addr, a.name)
  end

  for a in sorted
    dict[a] = Seep.llvm_val(a, mod, builder, eltype, map(x->dict[x], a.input)...)
    if haskey(outptrs, a)
      addr = LLVM.BuildInBoundsGEP(builder, outptrs[a], [index], string(a.name, "_out[index]"))
      LLVM.BuildStore(builder, dict[a], addr)
    end
  end
  next_index = LLVM.BuildAdd(builder, index, stride, "next_index")
  LLVM.AddIncoming(index, [next_index], [loop_block])
  LLVM.BuildBr(builder, test_block)

  # exit:
  LLVM.PositionBuilderAtEnd(builder, exit_block)
  LLVM.BuildRetVoid(builder)

  LLVM.dispose(builder)
end

const module_counter = [0]
function create_module()
  module_counter[1] += 1
  module_name = "SeepLlvmModule$(module_counter[1])"
  mod = LLVM.ModuleRef(module_name)

  LLVM.SetTarget(mod, "nvptx$(8*sizeof(Int))-nvidia-cuda")
  LLVM.SetDataLayout(mod, join(["e", "p:64:64:64",
    "i1:8:8", "i8:8:8", "i16:16:16", "i32:32:32", "i64:64:64",
    "f32:32:32", "f64:64:64",
    "v16:16:16", "v32:32:32", "v64:64:64", "v128:128:128",
    "n16:32:64"], "-"))

  return mod
end

function get_bitcode(mod::ModuleRef)
  if false
    return mktempdir() do tempdir
      t = joinpath(tempdir, "bitcode.bc")
      LLVM.WriteBitcodeToFile(mod, t)
      bitcode = open(readbytes, t)
      rm(t)
      bitcode
    end
  else
    t = "bitcode.bc"
    LLVM.WriteBitcodeToFile(mod, t)
    run(`llvm-dis $t`)
    bitcode = open(readbytes, "bitcode.ll")
    return bitcode
  end
end

function get_ptx(bitcode)
  p = NVVM.Program()
  NVVM.errcheck(NVVM.AddModuleToProgram(p, open(readbytes, "/opt/cuda/nvvm/libdevice/libdevice.compute_20.10.bc"), "libdevice"))
  NVVM.errcheck(NVVM.AddModuleToProgram(p, bitcode, "seep"))
  ret = NVVM.CompileProgram(p, ASCIIString[])
  log = NVVM.GetProgramLog(p)
  if log != "\0"
    info(log)
  end

  if ret != 0
    error(bytestring(NVVM.GetErrorString(ret)))
  end

  open(io->write(io, bytestring(NVVM.GetCompiledResult(p))), "bitcode.ptx", "w")

  ptx = bytestring(NVVM.GetCompiledResult(p))
  NVVM.DestroyProgram([p])
  return ptx
end

function check_ready(readyvec::Set{ANode}, readynovec::Set{ANode}, live::Set{ANode}, inputs, nodes)
  for node in nodes
    if node in live continue end
    if all(x->x in live, inputs[node])
      push!(elementwise(node) ? readyvec : readynovec, node)
    end
  end
end

function check_dead(live::Set{ANode}, outputs, nodes)
  for node in nodes
    if any(x->!(x in live), outputs[node]) continue end
    delete!(live, node)
  end
end

function llvm_instance(pool::StoragePool, nodes::Vector{ANode})
  T = atype(pool)
  stored_nodes = Set{ANode}(nodes)

  unsorted = connected_nodes(nodes)
  live = Set{ANode}()
  storage = Dict{ANode,Any}()
  userstorage = Dict{ANode,Any}()
  refcnt = Dict{ANode,RefCounter}()

  outputs,inputs = output_map(unsorted)

  for n in nodes
    if sym(n) == :noop continue end
    incr(refcnt, output_node(n))
  end

  constants = ANode[]
  c_block = :(begin end)

  for n in unsorted
    for b in n.input
      incr(refcnt, output_node(b))
    end

    if !elementwise(n)
      push!(stored_nodes, n)
      union!(stored_nodes, n.input)
    end

    if sym(n) == :input
      push!(live, n)
      userstorage[n] = storage[n] = allocate(pool, n.size...)
    end

    if sym(n) == :load && isa(arg(n), T)
      storage[n] = arg(n)
      push!(live, n)
    end

    if sym(n) == :const
      push!(live, n)
      if isa(arg(n), T)
        storage[n] = arg(n)
      else
        incr(refcnt, n)
        t = storage[n] = allocate(pool, n.size...)
        push!(c_block.args, :(copy!($t, map($(eltype(T)), $(arg(n))))))
        push!(constants, n)
      end
    end
  end

  @gensym initialize_constants
  @eval $(initialize_constants)() = $c_block

  readyvec = Set{ANode}()
  readynovec = Set{ANode}()
  check_ready(readyvec, readynovec, live, inputs, unsorted)

  block = :(begin end)
  if !isempty(constants)
    block = :(begin
      if $(pool).token[1] != $(QuoteNode(initialize_constants))
        $(pool).token[1] = $(QuoteNode(initialize_constants))
        $(initialize_constants)()
      end
    end)
  end

  mod = create_module()
  function_counter = 0

  node_counter = length(live)
  push!(block.args, :(device_synchronize()))
  need_sync = false
  free_nodes = ANode[]

  streams = [CuStream(1) for i in 1:50]
  current_stream = CUDAdrv.default_stream()
  while true
    #println("live: " * join(sort(collect(Int, map(x->x.id, live))), ", "))
    #println("readyvec: " * join(sort(collect(Int, map(x->x.id, readyvec))), ", "))
    #println("readynovec: " * join(sort(collect(Int, map(x->x.id, readynovec))), ", "))

    if !isempty(readynovec)
      ready = collect(readynovec)
      empty!(readynovec)
      i = 1
      #@show node.id

      if need_sync
        push!(block.args, :(device_synchronize()))
        need_sync = false
        for aa in free_nodes
          deallocate(pool, storage[aa])
        end
        empty!(free_nodes)
      end

      sort!(ready, by=length, rev=true)
      for node in ready
        if sym(node) == :noop continue end
        input = map(x->storage[x], node.input)

        if current_stream != streams[(i-1)%50+1] || true
          current_stream = streams[(i-1)%50+1]
          push!(block.args, :(STREAM[1] = $current_stream))
          push!(block.args, :(CUBLAS.cublasSetStream_v2(CUBLAS.cublashandle[1], STREAM[1].handle)))
        end

        if sym(node) == :input || sym(node) == :const
          error("oops");
        elseif sym(node) == :load
          if !haskey(storage, node)
            storage[node] = allocate(pool, node.size...)
          end
          push!(block.args, :(copy!($(storage[node]), arg($node), stream=$current_stream)))
        elseif mutates(node)
          storage[node] = input[1]
          push!(block.args, :(do_forward!($node, $(input...))))
        else
          if !haskey(storage, node)
            storage[node] = allocate(pool, node.size...)
          end
          push!(block.args, :(do_forward!($node, $(storage[node]), $(input...))))
        end


        for a in node.input
          aa = output_node(a)
          if decr(refcnt, aa) && arg(aa) != storage[aa]
            push!(free_nodes, aa)
          end
        end
        i += 1
      end

      if length(ready) > 1
        need_sync = true
      end

      for node in ready
        push!(live, node)
        check_ready(readyvec, readynovec, live, inputs, outputs[node])
        check_dead(live, outputs, inputs[node])
        node_counter += 1
      end
    elseif !isempty(readyvec)
      vnodes = ANode[]
      out_nodes = Set{ANode}()

      maxlen = maximum(length, readyvec)
      while true
        a = collect(filter(x->length(x) == maxlen, readyvec))
        if isempty(a) break end

        setdiff!(readyvec, a)
        append!(vnodes, a)
        union!(live, a)
        for node in a
          check_ready(readyvec, readynovec, live, inputs, outputs[node])
          if node in stored_nodes
            push!(out_nodes, node)
            if !haskey(storage, node)
              storage[node] = allocate(pool, node.size...)
            end
          end
          check_dead(live, outputs, node.input)

          for i in node.input
            aa = output_node(i)
            if decr(refcnt, aa) && aa in stored_nodes && arg(aa) != storage[aa]
              push!(free_nodes, aa)
            end
          end

          node_counter += 1
        end
      end

      for node in vnodes
        if !iszero(refcnt[node])
          push!(out_nodes, node)
          push!(stored_nodes, node)
          if !haskey(storage, node)
            storage[node] = allocate(pool, node.size...)
          end
        end
        @assert length(node) == length(vnodes[1])
      end

      ccs = connected_components(vnodes, inputs, outputs)
      opcount(cc::Vector{ANode}) = sum(length, cc)
      order = Base.Order.By(opcount)
      Collections.heapify!(ccs, order)

      while length(ccs) > 1
        a = Collections.heappop!(ccs, order)
        b = Collections.heappop!(ccs, order)
        if opcount(a) + opcount(b) < 1024*1024
          append!(a, b)
          Collections.heappush!(ccs, a, order)
        else
          push!(ccs, a)
          push!(ccs, b)
          break
        end
      end

      sort!(ccs, by=opcount, rev=true)

      if need_sync
        push!(block.args, :(device_synchronize()))
        need_sync = false
        for aa in free_nodes
          deallocate(pool, storage[aa])
        end
        empty!(free_nodes)
      end

      i = 1
      for cc in ccs
        function_counter += 1
        function_name = "function_$function_counter"
        in_nodes = Set{ANode}()
        for c in cc
          union!(in_nodes, c.input)
        end
        setdiff!(in_nodes, cc)

        in_node_vec = collect(ANode, in_nodes)
        out_node_vec = collect(ANode, filter(c->c in out_nodes, cc))

        args = [length(cc[1]); map(x->storage[output_node(x)], [out_node_vec; in_node_vec])]
        arg_types = (Int32, fill(Ptr{eltype(T)}, length(args)-1)...)
        build_grid_stride_loop(mod, function_name, eltype(T), out_node_vec, in_node_vec)
        push!(block.args, :(cudacall(g._fun[$function_counter], 256, 128, $arg_types, $(args...); stream=$(streams[(i-1)%50+1]))))
        i += 1
      end

      if length(ccs) > 1
        need_sync = true
      end
    else
      break
    end
  end

  current_stream = CUDAdrv.default_stream()
  push!(block.args, :(STREAM[1] = $current_stream))
  push!(block.args, :(CUBLAS.cublasSetStream_v2(CUBLAS.cublashandle[1], STREAM[1].handle)))
  push!(block.args, :(device_synchronize()))

  for aa in free_nodes
    deallocate(pool, storage[aa])
  end

  @assert node_counter == length(unsorted)

  for n in nodes
    if sym(n) == :noop continue end
    aa = output_node(n)
    userstorage[n] = storage[aa]
    if decr(refcnt, aa) && arg(aa) != storage[aa]
      deallocate(pool, storage[aa])
    end
  end

  for i in constants
    if decr(refcnt, i)
      deallocate(pool, storage[i])
    end
  end

  @assert all(iszero, values(refcnt))

  bitcode = get_bitcode(mod)
  LLVM.dispose(mod)

  info("Compiling to PTX")
  ptx = @time get_ptx(bitcode)

  m = CuModuleData(ptx)

  @gensym SeepLlvmInstance
  type_block = quote
    immutable $SeepLlvmInstance <: Instance
      _mod::CuModule
      _fun::Vector{CuFunction}
      _stream::Vector{CuStream}
      _pool::StoragePool
      _storage::Dict{ANode,$T}
    end
    Base.show(io::IO, ::$SeepLlvmInstance) = print(io, "Seep LLVM Instance with $($(length(unsorted))) nodes.")
    @compat (g::$SeepLlvmInstance)() = $block
    $SeepLlvmInstance($m, $(CuFunction[CuFunction(m, "function_$i") for i in 1:function_counter]), $streams, $pool, $userstorage,)
  end

  for (n,a) in userstorage
    if haskey(userstorage, n) && n.name != ""
      if length(filter(x->x.name == n.name, unsorted)) > 1
        warn("There are multiple nodes named $(n.name).")
        continue
      end
      push!(type_block.args[2].args[3].args, :($(symbol(n.name))::$(typeof(a))))
      push!(type_block.args[end].args, a)
    end
  end

  info("Evaluating type")
  return @time eval(type_block)
end

