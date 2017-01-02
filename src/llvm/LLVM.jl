# © 2016 Massachusetts Institute of Technology.  See LICENSE file for details.
__precompile__(false)
module LLVM

function __init__()
  global const llvmcore = Libdl.dlopen("libLLVMCore.so")
  global const llvmbitwriter = Libdl.dlopen("libLLVMBitWriter.so")

  for i in check_syms
    Libdl.dlsym(eval(i[1]), i[2])
  end
end

lib = :llvmcore
fun_prefix = "LLVM"

for t in [:Context, :MemoryBuffer, :Module, :Type, :Value, :BasicBlock, :Use, :Builder, :Message]
  tt = symbol(t, "Ref")
  @eval immutable $tt ptr::Ptr{Void} end
end

include("fun.jl")

macro aget(getter, lengetter, intype, outtype)
  quote
    @fun $(esc(lengetter))(v::$intype)::Cuint
    @fun $(esc(getter))(v::$intype, a::Vector{$outtype})::Void = (v, a::Ptr{$outtype})
    function $(esc(getter))(v::$intype)
      a = Array($outtype, $lengetter(v))
      $(esc(getter))(v, a)
      return a
    end
  end
end

macro iter(suffix, t1, t2)
  quote
    @fun $(esc(symbol("GetFirst", suffix)))(v::$t2)::$t1
    @fun $(esc(symbol("GetLast", suffix)))(v::$t2)::$t1
    @fun $(esc(symbol("GetNext", suffix)))(v::$t1)::$t1
    @fun $(esc(symbol("GetPrevious", suffix)))(v::$t1)::$t1
  end
end

# Core
#@fun CreateMessage(msg::AbstractString)::MessageRef = (msg::Cstring,)
#@fun DisposeMessage(msg::MessageRef)::Void
#dispose(msg::MessageRef) = DisposeMessage(msg)
#Base.bytestring(msg::MessageRef) = bytestring(reinterpret(Ptr{UInt8}, msg.ptr))
#Base.string(msg::MessageRef) = bytestring(msg)

# Context
@fun ContextCreate()::ContextRef
@fun GetGlobalContext()::ContextRef
@fun ContextDispose(c::ContextRef)::Void
@fun GetMDKindIDInContext(name::AbstractString)::Cuint = (name::Cstring, length(name)::Cuint)
@fun GetMDKindIDInContext(c::ContextRef, name::AbstractString)::Cuint = (c, name::Cstring, length(name)::Cuint)

ContextRef() = ContextCreate()
dispose(c::ContextRef) = ContextDispose(c)

# Memory Buffer
#@fun GetBufferStart(m::MemoryBufferRef)::Ptr{UInt8}
#@fun GetBufferSize(m::MemoryBufferRef)::Csize_t
#@fun DisposeMemoryBuffer(m::MemoryBufferRef)::Void
#dispose(m::MemoryBufferRef) = DisposeMemoryBuffer(m)
#function getbuffer(m::MemoryBufferRef)
#  data = copy(pointer_to_array(GetBufferStart(m), GetBufferSize(m)))
#  DisposeMemoryBuffer(m)
#  return data
#end

# Metadata
@fun MDString(s::AbstractString)::ValueRef = (s::Cstring, length(s)::Cuint)
@fun MDStringInContext(c::ContextRef, s::AbstractString)::ValueRef = (c, s::Cstring, length(s)::Cuint)
@fun MDNode(vals::Vector{ValueRef})::ValueRef = (vals::Ptr{ValueRef}, length(vals)::Cuint)
@fun MDNodeInContext(c::ContextRef, vals::Vector{ValueRef})::ValueRef = (c, vals::Ptr{ValueRef}, length(vals)::Cuint)
@fun GetMDString(v::ValueRef, len::Vector{Cuint})::Cstring = (v, len::Ptr{Cuint})
@aget GetMDNodeOperands GetMDNodeNumOperands ValueRef ValueRef

# Module
@fun ModuleCreateWithName(s::AbstractString)::ModuleRef = (s::Cstring,)
@fun ModuleCreateWithNameInContext(c::ContextRef, s::AbstractString)::ModuleRef = (c, s::Cstring,)
#@fun CloneModule(m::ModuleRef)::ModuleRef
@fun DisposeModule(m::ModuleRef)::Void
@fun AddFunction(mod::ModuleRef, name::AbstractString, llvmtype::TypeRef)::ValueRef = (mod, name::Cstring, llvmtype)
@fun GetNamedFunction(mod::ModuleRef, name::AbstractString)::ValueRef = (mod, name::Cstring)
@iter Function ValueRef ModuleRef
@fun DumpModule(m::ModuleRef)::Void
#@fun PrintModuleToString(m::ModuleRef)::MessageRef

#Base.string(m::ModuleRef) = bytestring(PrintModuleToString(m))

@fun GetDataLayout(m::ModuleRef)::Cstring
@fun SetDataLayout(m::ModuleRef, layout::AbstractString)::Void = (m, layout::Cstring)
@fun GetTarget(m::ModuleRef)::Cstring
@fun SetTarget(m::ModuleRef, triple::AbstractString)::Void = (m, triple::Cstring)

SetDataLayout(m::ModuleRef, layout::AbstractArray) = SetDataLayout(m, join(layout, "-"))

@fun GetNamedMetadataNumOperands(m::ModuleRef, name::AbstractString)::Cuint = (m, name::Cstring)
@fun GetNamedMetadataOperands(m::ModuleRef, name::AbstractString, out::Vector{ValueRef})::Void = (m, name::Cstring, out::Ptr{ValueRef})
@fun AddNamedMetadataOperand(m::ModuleRef, name::AbstractString, val::ValueRef)::Void = (m, name::Cstring, val)

ModuleRef(name::AbstractString) = ModuleCreateWithName(name)
ModuleRef(c::ContextRef, name::AbstractString) = ModuleCreateWithNameInContext(c, name)
#Base.copy(m::ModuleRef) = CloneModule(m)
dispose(m::ModuleRef) = DisposeModule(m)

lib = :llvmbitwriter
#@fun WriteBitcodeToMemoryBuffer(m::ModuleRef)::MemoryBufferRef
@fun WriteBitcodeToFile(m::ModuleRef, path::AbstractString)::MemoryBufferRef = (m, path::Cstring)

lib = :llvmcore
# Type

## Float, Int, and Other types

@fun IntType(width::Integer)::ValueRef = (width::Cuint,)
@fun IntTypeInContext(c::ContextRef, width::Integer)::ValueRef = (c, width::Cuint)
@fun GetIntTypeWidth(t::TypeRef)::Cint

for t in [8,16,32,64] # 128
  @eval @fun $(symbol("Int", t, "Type"))()::TypeRef
  @eval @fun $(symbol("Int", t, "TypeInContext"))(c::ContextRef)::TypeRef

  @eval TypeRef(::Type{$(symbol("Int", t))}) = $(symbol("Int", t, "Type"))()
  @eval Base.convert(::Type{TypeRef}, ::Type{$(symbol("Int", t))}) = $(symbol("Int", t, "Type"))()

  @eval TypeRef(::Type{$(symbol("UInt", t))}) = $(symbol("Int", t, "Type"))()
  @eval Base.convert(::Type{TypeRef}, ::Type{$(symbol("UInt", t))}) = $(symbol("Int", t, "Type"))()
end

for t in [:Int1, :Void, :Label, :X86MMX, :Half, :Float, :Double, :X86FP80, :FP128, :PPCFP128]
  @eval @fun $(symbol(t, "Type"))()::TypeRef
  @eval @fun $(symbol(t, "TypeInContext"))(c::ContextRef)::TypeRef

  if isdefined(Base, t) && isa(eval(t), Type)
    @eval TypeRef(::Type{$t}) = $(symbol(t, "Type"))()
    @eval Base.convert(::Type{TypeRef}, ::Type{$t}) = $(symbol(t, "Type"))()
  end
end

for (t,j) in (:Int1=>:Bool, :Half=>:Float16, :Float=>:Float32, :Double=>:Float64)
  @eval TypeRef(::Type{$j}) = $(symbol(t, "Type"))()
  @eval Base.convert(::Type{TypeRef}, ::Type{$j}) = $(symbol(t, "Type"))()
end

## Function Types
@fun FunctionType(ret::TypeRef, param::Vector{TypeRef}, var::Bool=false)::TypeRef =
  (ret::TypeRef, param::Ptr{TypeRef}, length(param)::Cuint, (var ? 1 : 0)::Cint)

@fun IsFunctionVarArg(f::TypeRef)::Cint
@fun GetReturnType(f::TypeRef)::TypeRef
@aget GetParamTypes CountParamTypes TypeRef TypeRef
@aget GetParams CountParams ValueRef ValueRef
@fun GetParam(fn::ValueRef, i::Integer)::ValueRef = (fn, i::Cuint)
@fun GetParamParent(val::ValueRef)::ValueRef
@iter Param ValueRef ValueRef


## Sequence Types
@fun ArrayType(f::TypeRef, len::Integer)::TypeRef = (f, len::Cuint)
@fun GetElementType(t::TypeRef)::TypeRef
@fun GetArrayLength(f::TypeRef)::Cuint
@fun PointerType(f::TypeRef, space::Integer)::TypeRef = (f::TypeRef, space::Cuint)
@fun GetPointerAddressSpace(f::TypeRef)::Cuint
@fun VectorType(f::TypeRef, len::Integer)::Cuint = (f, len::Cuint)
@fun GetVectorSize(f::TypeRef)::Cuint

## Structure Types
@fun StructType(eltypes::Vector{TypeRef}, packed::Bool)::TypeRef = (eltypes::Ptr{TypeRef}, length(eltypes)::Cuint, (packed?1:0)::Cuint)
@fun StructTypeInContext(c::ContextRef, eltypes::Vector{TypeRef}, packed::Bool)::TypeRef = (c, eltypes::Ptr{TypeRef}, length(eltypes)::Cuint, (packed?1:0)::Cuint)
@fun StructCreateNamed(c::ContextRef, name::AbstractString)::TypeRef = (c, name::Cstring)
@fun GetStructName(t::TypeRef)::Cstring
@fun StructSetBody(t::TypeRef, eltypes::Vector{TypeRef}, packed::Bool)::Void = (t, eltypes::Ptr{TypeRef}, length(eltypes)::Cuint, (packed?1:0)::Cuint)
#@fun StructGetTypeAtIndex(t::TypeRef, i::Cuint)::TypeRef
@fun IsPackedStruct(t::TypeRef)::Cuint
@fun IsOpaqueStruct(t::TypeRef)::Cuint

@aget GetStructElementTypes CountStructElementTypes TypeRef TypeRef

# Value
@fun TypeOf(val::ValueRef)::TypeRef
@fun GetValueName(val::ValueRef)::Cstring
@fun SetValueName(val::ValueRef, name::AbstractString)::Void = (val, name::Cstring)
@fun DumpValue(val::ValueRef)::Void
#@fun PrintValueToString(val::ValueRef)::MessageRef
@fun ReplaceAllUsesWith(old::ValueRef, new::ValueRef)::Void
@fun IsConstant(v::ValueRef)::Cuint
@fun IsUndef(v::ValueRef)::Cuint
@fun IsAMDNode(v::ValueRef)::ValueRef
@fun IsAMDString(v::ValueRef)::ValueRef

@fun GetFirstUse(val::ValueRef)::UseRef
@fun GetNextUse(val::UseRef)::UseRef
@fun GetUser(u::UseRef)::ValueRef
@fun GetUsedValue(u::UseRef)::ValueRef

@fun GetOperand(val::ValueRef, index::Integer)::ValueRef = (val, index::Cuint)
#@fun GetOperandUse(val::ValueRef, index::Integer)::UseRef = (val, index::Cuint)
@fun SetOperand(user::ValueRef, index::Integer, val::ValueRef)::Void = (user, index::Cuint, val)
@fun GetNumOperands(val::ValueRef)::Cint

@fun ConstNull(ty::TypeRef)::ValueRef
@fun ConstAllOnes(ty::TypeRef)::ValueRef
@fun GetUndef(ty::TypeRef)::ValueRef
@fun IsNull(val::ValueRef)::Cuint
@fun ConstPointerNull(ty::TypeRef)::ValueRef

@fun ConstInt(t::TypeRef, n::Integer, sext::Bool)::ValueRef =
  (t, (typeof(n) <: Signed ? reinterpret(Culonglong, Clonglong(n)) : Culonglong(n))::Culonglong, sext::Cuint)
@fun ConstReal(t::TypeRef, n::Real)::ValueRef = (t, n::Cdouble)
@fun ConstIntGetZExtValue(v::ValueRef)::Culonglong
@fun ConstIntGetSExtValue(v::ValueRef)::Clonglong
#@fun ConstRealGetDouble(v::ValueRef, losesInfo::Vector{Cuint})::Cdouble = (v,losesInfo::Ptr{Cuint})

@fun AddFunctionAttr(fn::ValueRef, a::Integer)::Void = (fn, a::Cuint)
@fun GetFunctionAttr(fn::ValueRef)::Cuint
@fun RemoveFunctionAttr(fn::ValueRef, a::Integer)::Void = (fn, a::Cuint)

const ZExtAttribute       = 1<<0
const SExtAttribute       = 1<<1
const NoReturnAttribute   = 1<<2
const InRegAttribute      = 1<<3
const StructRetAttribute  = 1<<4
const NoUnwindAttribute   = 1<<5
const NoAliasAttribute    = 1<<6
const ByValAttribute      = 1<<7
const NestAttribute       = 1<<8
const ReadNoneAttribute   = 1<<9
const ReadOnlyAttribute   = 1<<10
const NoInlineAttribute   = 1<<11
const AlwaysInlineAttribute    = 1<<12
const OptimizeForSizeAttribute = 1<<13
const StackProtectAttribute    = 1<<14
const StackProtectReqAttribute = 1<<15
const Alignment = 31<<16
const NoCaptureAttribute  = 1<<21
const NoRedZoneAttribute  = 1<<22
const NoImplicitFloatAttribute = 1<<23
const NakedAttribute      = 1<<24
const InlineHintAttribute = 1<<25
const StackAlignment = 7<<26
const ReturnsTwice = 1 << 29
const UWTable = 1 << 30
const NonLazyBind = 1 << 31

# Basic Block
@fun BasicBlockAsValue(bb::BasicBlockRef)::ValueRef
@fun ValueIsBasicBlock(v::ValueRef)::Cuint
@fun ValueAsBasicBlock(v::ValueRef)::BasicBlockRef
@fun GetBasicBlockParent(bb::BasicBlockRef)::ValueRef
@fun GetBasicBlockTerminator(bb::BasicBlockRef)::ValueRef
@aget GetBasicBlocks CountBasicBlocks ValueRef BasicBlockRef
@iter BasicBlock BasicBlockRef ValueRef
@fun GetEntryBasicBlock(fn::ValueRef)::BasicBlockRef
@fun AppendBasicBlock(fn::ValueRef, name::AbstractString)::BasicBlockRef = (fn, name::Cstring)
@fun AppendBasicBlockInContext(c::ContextRef, fn::ValueRef, name::AbstractString)::BasicBlockRef = (c, fn, name::Cstring)
@fun InsertBasicBlock(bb::BasicBlockRef, name::AbstractString)::BasicBlockRef = (bb, name::Cstring)
@fun InsertBasicBlockInContext(c::ContextRef, bb::BasicBlockRef, name::AbstractString)::BasicBlockRef = (c, bb, name::Cstring)
@fun DeleteBasicBlock(bb::BasicBlockRef)::Void
@fun RemoveBasicBlockFromParent(bb::BasicBlockRef)::Void
@fun MoveBasicBlockBefore(bb::BasicBlockRef, pos::BasicBlockRef)::Void
@fun MoveBasicBlockAfter(bb::BasicBlockRef, pos::BasicBlockRef)::Void
@iter Instruction ValueRef BasicBlockRef

# InstructionBuilder
@fun CreateBuilder()::BuilderRef
@fun CreateBuilderInContext(c::ContextRef)::BuilderRef
@fun PositionBuilder(b::BuilderRef, bb::BasicBlockRef, val::ValueRef)::Void
@fun PositionBuilderBefore(b::BuilderRef, val::ValueRef)::Void
@fun PositionBuilderAtEnd(b::BuilderRef, val::BasicBlockRef)::Void
@fun GetInsertBlock(b::BuilderRef)::BasicBlockRef
@fun ClearInsertionPosition(b::BuilderRef)::Void
@fun InsertIntoBuilder(b::BuilderRef, instr::ValueRef)::Void
@fun InsertIntoBuilderWithName(b::BuilderRef, instr::ValueRef, name::AbstractString)::Void = (b, instr, name::Cstring)
@fun DisposeBuilder(b::BuilderRef)::Void
dispose(b::BuilderRef) = DisposeBuilder(b)

for i in ["RetVoid", "Unreachable"]
  @eval @fun $(symbol("Build", i))(b::BuilderRef)::ValueRef
end

for i in ["Ret", "Resume", "Free"]
  @eval @fun $(symbol("Build", i))(b::BuilderRef, v::ValueRef)::ValueRef
end

for i in ["Add", "NSWAdd", "NUWAdd", "FAdd",
          "Sub", "NSWSub", "NUWSub", "FSub",
          "Mul", "NSWMul", "NUWMul", "FMul",
          "UDiv", "SDiv", "ExactSDiv", "FDiv",
          "URem", "SRem", "FRem",
          "Shl", "LShr", "AShr",
          "And", "Or", "Xor",
          "Store", "PtrDiff"]
  @eval @fun $(symbol("Build", i))(b::BuilderRef, lhs::ValueRef, rhs::ValueRef, name::AbstractString)::ValueRef = (b, lhs, rhs, name::Cstring)
end

for i in ["Neg", "NSWNeg", "NUWNeg", "FNeg", "Not", "Load", "IsNull", "IsNotNull", "Load"]
  @eval @fun $(symbol("Build", i))(b::BuilderRef, val::ValueRef, name::AbstractString)::ValueRef = (b, val, name::Cstring)
end

for i in ["ICmp", "FCmp"]
  @eval @fun $(symbol("Build", i))(b::BuilderRef, pred::Integer, lhs::ValueRef, rhs::ValueRef, name::AbstractString)::ValueRef = (b, pred::Cint, lhs, rhs, name::Cstring)
end

const IntEQ = 32      # equal
const IntNE = 33      # not equal
const IntUGT = 34     # unsigned greater than
const IntUGE = 35     # unsigned greater or equal
const IntULT = 36     # unsigned less than
const IntULE = 37     # unsigned less or equal
const IntSGT = 38     # signed greater than
const IntSGE = 39     # signed greater or equal
const IntSLT = 40     # signed less than
const IntSLE = 41      # signed less or equal

const RealPredicateFalse = 0   # Always false (always folded)
const RealOEQ = 1              # True if ordered and equal
const RealOGT = 2              # True if ordered and greater than
const RealOGE = 3              # True if ordered and greater than or equal
const RealOLT = 4              # True if ordered and less than
const RealOLE = 5              # True if ordered and less than or equal
const RealONE = 6              # True if ordered and operands are unequal
const RealORD = 7              # True if ordered (no nans)
const RealUNO = 8              # True if unordered: isnan(X) | isnan(Y)
const RealUEQ = 9              # True if unordered or equal
const RealUGT = 10             # True if unordered or greater than
const RealUGE = 11             # True if unordered, greater than, or equal
const RealULT = 12             # True if unordered or less than
const RealULE = 13             # True if unordered, less than, or equal
const RealUNE = 14             # True if unordered or not equal
const RealPredicateTrue = 15   # Always true (always folded)

@fun BuildStore(b::BuilderRef, val::ValueRef, ptr::ValueRef)::ValueRef
@fun BuildCall(b::BuilderRef, fn::ValueRef, args::Vector{ValueRef}, name::AbstractString)::ValueRef = (b, fn, args::Ptr{ValueRef}, length(args)::Cuint, name::Cstring)
@fun BuildGEP(b::BuilderRef, p::ValueRef, inds::Vector{ValueRef}, name::AbstractString)::ValueRef = (b, p, inds::Ptr{ValueRef}, length(inds)::Cuint, name::Cstring)
@fun BuildInBoundsGEP(b::BuilderRef, p::ValueRef, inds::Vector{ValueRef}, name::AbstractString)::ValueRef = (b, p, inds::Ptr{ValueRef}, length(inds)::Cuint, name::Cstring)

@fun BuildBr(b::BuilderRef, dest::BasicBlockRef)::ValueRef
@fun BuildCondBr(b::BuilderRef, cond::ValueRef, then::BasicBlockRef, otherwise::BasicBlockRef)::ValueRef

@fun BuildPhi(b::BuilderRef, ty::TypeRef, name::AbstractString)::ValueRef = (b, ty, name::Cstring)
@fun AddIncoming(phi::ValueRef, vals::Vector{ValueRef}, blocks::Vector{BasicBlockRef})::Void = (phi, vals::Ptr{ValueRef}, blocks::Ptr{BasicBlockRef}, (@assert length(vals) == length(blocks); length(vals))::Cuint)
@fun CountIncoming(phi::ValueRef)::Cuint
@fun GetIncomingValue(phi::ValueRef, index::Integer)::ValueRef = (phi, index::Cuint)
@fun GetIncomingBlock(phi::ValueRef, index::Integer)::BasicBlockRef = (phi, index::Cuint)

end
