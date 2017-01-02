# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
module NVVM

function __init__()
  global const libnvvm = Libdl.dlopen(Libdl.find_library(["libnvvm.so.2"], ["/opt/cuda/nvvm/lib64"]))

  for i in check_syms
    Libdl.dlsym(eval(i[1]), i[2])
  end
end

include("fun.jl")

lib = :libnvvm
fun_prefix = "nvvm"

immutable Program p::Ptr{Void} end
const SUCCESS = 0
const ERROR_OUT_OF_MEMORY = 1
const ERROR_PROGRAM_CREATION_FAILURE = 2
const ERROR_IR_VERSION_MISMATCH = 3
const ERROR_INVALID_INPUT = 4
const ERROR_INVALID_PROGRAM = 5
const ERROR_INVALID_IR = 6
const ERROR_INVALID_OPTION = 7
const ERROR_NO_MODULE_IN_PROGRAM = 8
const ERROR_COMPILATION = 9

"Get the message string for the given nvvmResult code"
@fun GetErrorString(result::Cint)::Cstring
errcheck(result::Cint) = result == SUCCESS ? nothing : error(bytestring(GetErrorString(result)))

"Create a program, and set the value of its handle to *prog"
#nvvmResult nvvmCreateProgram ( nvvmProgram* prog )
@fun CreateProgram(prog::Vector{Program})::Cint = (prog::Ptr{Program},)

function Program()
  prog = Array(Program, 1)
  errcheck(CreateProgram(prog))
  return prog[1]
end

"Destroy a program"
#nvvmResult nvvmDestroyProgram ( nvvmProgram* prog )
@fun DestroyProgram(prog::Vector{Program})::Cint = (prog::Ptr{Program},)
dispose(prog::Program) = errcheck(DestroyProgram([prog]))

"Add a module level NVVM IR to a program"
#nvvmResult nvvmAddModuleToProgram ( nvvmProgram prog, const char* buffer, size_t size, const char* name )
@fun AddModuleToProgram(prog::Program, buffer::Array{UInt8}, name::AbstractString)::Cint =
  (prog, buffer::Ptr{UInt8}, length(buffer)::Csize_t, name::Cstring)

"Compile the NVVM program"
#nvvmResult nvvmCompileProgram ( nvvmProgram prog, int  numOptions, const char** options )
@fun CompileProgram(prog::Program, options::Vector{ASCIIString})::Cint =
  (prog, length(options)::Cint, map(Cstring, options)::Ptr{Cstring})

"Get the compiled result"
#nvvmResult nvvmGetCompiledResult ( nvvmProgram prog, char* buffer )
@fun GetCompiledResult(prog::Program, buffer::Vector{UInt8})::Cint = (prog, buffer::Ptr{UInt8})

"Get the size of the compiled result"
#nvvmResult nvvmGetCompiledResultSize ( nvvmProgram prog, size_t* bufferSizeRet )
@fun GetCompiledResultSize(prog::Program, buffer::Vector{Csize_t})::Cint = (prog, buffer::Ptr{Csize_t})

function GetCompiledResult(prog::Program)
  sz = Csize_t[0]
  errcheck(GetCompiledResultSize(prog, sz))
  buf = Array(UInt8, sz[1])
  errcheck(GetCompiledResult(prog, buf))
  return buf
end

"Get the Compiler/Verifier Message"
#nvvmResult nvvmGetProgramLog ( nvvmProgram prog, char* buffer )
@fun GetProgramLog(prog::Program, buffer::Vector{UInt8})::Cint = (prog, buffer::Ptr{UInt8})

"Get the Size of Compiler/Verifier Message"
#nvvmResult nvvmGetProgramLogSize ( nvvmProgram prog, size_t* bufferSizeRet )
@fun GetProgramLogSize(prog::Program, buffer::Vector{Csize_t})::Cint = (prog, buffer::Ptr{Csize_t})

function GetProgramLog(prog::Program)
  sz = Csize_t[0]
  errcheck(GetProgramLogSize(prog, sz))
  buf = Array(UInt8, sz[1])
  errcheck(GetProgramLog(prog, buf))
  return bytestring(buf)
end

"Verify the NVVM program"
#nvvmResult nvvmVerifyProgram ( nvvmProgram prog, int  numOptions, const char** options )
@fun VerifyProgram(prog::Program, options::Vector{ASCIIString})::Cint =
  (prog, length(options)::Cint, map(Cstring, options)::Ptr{Cstring})

end
