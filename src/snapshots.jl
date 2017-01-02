# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
using HDF5, JLD

typealias File Union{JLD.JldFile, HDF5.HDF5File}

# load/save filename
load_snapshot(filename::AbstractString, args...) = jldopen(jld->load_snapshot(jld, args...), filename, "r")

function save_snapshot(filename::AbstractString, args...)
  tmppath,tmpio = mktemp(dirname(filename))
  close(tmpio)
  try
    jldopen(jld->save_snapshot(jld, args...), tmppath, "w")
    Base.FS.rename(tmppath, filename)
    return
  catch
    Base.FS.unlink(tmppath)
    rethrow()
  end
end

# load/save ANode...
save_snapshot(file::File, n::ANode...) = save_snapshot(file, collect(ANode, n))
load_snapshot(file::File, n::ANode...) = load_snapshot(file, collect(ANode, n))

save_snapshot(file::AbstractString, n::ANode...) = save_snapshot(file, collect(ANode, n))
load_snapshot(file::AbstractString, n::ANode...) = load_snapshot(file, collect(ANode, n))

# load/save Instance
function load_snapshot(jld::File, g::Instance)
  for fn in fieldnames(g)
    if startswith(string(fn), "_") continue end
    copy!(g.(fn), read(jld, string(fn)))
  end
end

function save_snapshot(jld::File, g::Instance)
  for fn in fieldnames(g)
    if startswith(string(fn), "_") continue end
    if isdefined(:CudaArray) && isa(g.(fn), CudaArray)
      write(jld, string(fn), to_host(g.(fn)))
    else
      write(jld, string(fn), g.(fn))
    end
  end
end

# load/save Vector{ANode}
function load_snapshot{A<:ANode}(jld::File, ns::Vector{A}, visited=Set{ANode}())
  for n in ns
    load_snapshot(jld, n, visited)
  end
end

function save_snapshot{A<:ANode}(jld::File, ns::Vector{A}, visited=Set{ANode}())
  for n in ns
    save_snapshot(jld, n, visited)
  end
end

# load/save one ANode
function load_snapshot(jld::File, n::ANode, visited=Set{ANode}())
  if n in visited return end
  push!(visited, n)

  if !isempty(n.name) && sym(n) == :load
    try
      copy!(arg(n), read(jld, n.name))
    catch e
      error("Unable to load $(n.name): $e")
    end
  end

  for ni in n.input
    load_snapshot(jld, ni, visited)
  end
end

function save_snapshot(jld::File, n::ANode, visited=Set{ANode}())
  if n in visited return end
  push!(visited, n)

  if !isempty(n.name) && sym(n) == :load
    if isdefined(:CudaArray) && isa(n.arg, CudaArray)
      write(jld, n.name, to_host(n.arg))
    else
      write(jld, n.name, n.arg)
    end
  end

  for ni in n.input
    save_snapshot(jld, ni, visited)
  end
end
