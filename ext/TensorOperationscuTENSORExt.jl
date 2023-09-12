module TensorOperationscuTENSORExt

using TensorOperations
using cuTENSOR: cuTENSOR, CUDA, handle, CuTensorDescriptor, cudaDataType_t,
                cutensorContractionDescriptor_t,
                cutensorContractionFind_t, cutensorContractionPlan_t,
                CUTENSOR_OP_IDENTITY,
                CUTENSOR_OP_CONJ, CUTENSOR_OP_ADD, CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_WORKSPACE_RECOMMENDED, cutensorPermutation,
                cutensorElementwiseBinary, cutensorReduction,
                cutensorReductionGetWorkspace,
                cutensorComputeType, cutensorGetAlignmentRequirement,
                cutensorInitContractionDescriptor, cutensorInitContractionFind,
                cutensorContractionGetWorkspace, cutensorInitContractionPlan,
                cutensorContraction, CUTENSORError
using cuTENSOR.CUDA: CUDA, CuArray, AnyCuArray, with_workspace, default_stream
using cuTENSOR.CUDA.CUBLAS: CublasFloat, CublasReal
# this might be dependency-piracy, but removes a dependency from the main package
using cuTENSOR.CUDA.Adapt: adapt
using TensorOperations
using TupleTools
using Strided

const TO = TensorOperations
const CuStridedView{T,N,A} = StridedView{T,N,A} where {T,N,A<:CuArray{T}}
const StridedCUDA = TO.Backend{:StridedCUDA}

#-------------------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------------------

function TO.tensorscalar(C::AnyCuArray)
    return ndims(C) == 0 ? tensorscalar(collect(C)) : throw(DimensionMismatch())
end

@inline function tensorop(T::Type{<:Number}, op::Symbol)
    return (T <: Real || op == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
end

function cuTENSOR.CuTensorDescriptor(a::CuStridedView{T}; size=size(a), strides=strides(a),
                                     eltype=T) where {T}
    return cuTENSOR.CuTensorDescriptor(parent(a); size, strides, eltype,
                                       op=T <: Real ? CUTENSOR_OP_IDENTITY :
                                          _strideop_to_cuop(a.op))
end

_strideop_to_cuop(::typeof(identity)) = CUTENSOR_OP_IDENTITY
_strideop_to_cuop(::typeof(conj)) = CUTENSOR_OP_CONJ

# this is type piracy -> should be implemented in Strided.jl
function Base.unsafe_convert(::Type{CUDA.CuPtr{T}}, a::CuStridedView{T}) where {T}
    return pointer(a.parent, a.offset + 1)
end

function tensordescriptor(T::Type{<:Number}, A::CuArray, pA::Index2Tuple, conjA::Symbol)
    return CuTensorDescriptor(A;
                              op=tensorop(T, conjA),
                              size=TupleTools.getindices(size(A), linearize(pA)),
                              strides=TupleTools.getindices(strides(A), linearize(pA)))
end

#-------------------------------------------------------------------------------------------
# Operations
#-------------------------------------------------------------------------------------------

function TO.tensoradd!(C::AnyCuArray, pC::Index2Tuple, A::AnyCuArray, conjA::Symbol,
                       α::Number, β::Number)
    return tensoradd!(C, pC, A, conjA, α, β, StridedCUDA())
end
function TO.tensoradd!(C::AbstractArray, pC::Index2Tuple, A::AbstractArray, conjA::Symbol,
                       α::Number, β::Number, backend::StridedCUDA)
    tensoradd!(StridedView(C), pC, StridedView(A), conjA, α, β, backend)
    return C
end
function TO.tensoradd!(C::CuStridedView, pC::Index2Tuple, A::CuStridedView, conjA::Symbol,
                       α::Number, β::Number, ::StridedCUDA=StridedCUDA())
    # TODO: check if these checks are necessary
    # -- in principle cuTENSOR already checks this, but the error messages are less clear
    TO.argcheck_tensoradd(C, pC, A)
    TO.dimcheck_tensoradd(C, pC, A)

    TC = eltype(C)
    descC = CuTensorDescriptor(C)
    modeC = collect(Cint, linearize(pC))

    A′ = TO.flag2op(conjA)(A)
    descA = CuTensorDescriptor(A′)
    modeA = collect(Cint, 1:ndims(A′))

    T = convert(cudaDataType_t, TC)
    stream = default_stream()
    h = handle()

    if β == zero(β)
        cutensorPermutation(h, TC[α], A′, descA, modeA, C, descC,
                            modeC, T, stream)
    else
        cutensorElementwiseBinary(h, TC[α],
                                  A′, descA, modeA,
                                  TC[β],
                                  C, descC, modeC,
                                  C, descC, modeC,
                                  CUTENSOR_OP_ADD, T, stream)
    end

    return C
end

function TO.tensorcontract!(C::AnyCuArray, pC::Index2Tuple,
                            A::AnyCuArray, pA::Index2Tuple, conjA::Symbol,
                            B::AnyCuArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, StridedCUDA())
end
function TO.tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                            A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::StridedCUDA)
    tensorcontract!(StridedView(C), pC, StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB, α, β, backend)
    return C
end
function TO.tensorcontract!(C::CuStridedView{T}, pC::Index2Tuple,
                            A::CuStridedView, pA::Index2Tuple, conjA::Symbol,
                            B::CuStridedView, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, ::StridedCUDA=StridedCUDA()) where {T}
    # TODO: check if these checks are necessary
    # -- in principle cuTENSOR already checks this, but the error messages are less clear
    TO.argcheck_tensorcontract(C, pC, A, pA, B, pB)
    TO.dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    A′ = permutedims(TO.flag2op(conjA)(A), linearize(pA))
    B′ = permutedims(TO.flag2op(conjB)(B), linearize(pB))

    desc = _contraction_descriptor(C, pC, A′, pA, B′, pB)
    find = Ref{cutensorContractionFind_t}()
    cutensorInitContractionFind(handle(), find, CUTENSOR_ALGO_DEFAULT)

    function workspacesize()
        out = Ref{UInt64}(C_NULL)
        cutensorContractionGetWorkspace(handle(), desc, find,
                                        CUTENSOR_WORKSPACE_RECOMMENDED, out)
        return out[]
    end
    with_workspace(workspacesize, 1 << 27) do workspace
        plan_ref = Ref{cutensorContractionPlan_t}()
        cutensorInitContractionPlan(handle(), plan_ref, desc, find, sizeof(workspace))

        return cutensorContraction(handle(), plan_ref, T[α], A′, B′, T[β],
                                   C, C,
                                   workspace, sizeof(workspace), default_stream())
    end

    return C
end

function _contraction_descriptor(C::CuStridedView, pC::Index2Tuple, A::CuStridedView,
                                 pA::Index2Tuple, B::CuStridedView, pB::Index2Tuple)
    descC = CuTensorDescriptor(C)
    # pA′ = linearize(pA)
    descA = CuTensorDescriptor(A)
    #    size=TupleTools.getindices(size(A), pA′),
    #    strides=TupleTools.getindices(strides(A), pA′))
    # pB′ = linearize(pB)
    descB = CuTensorDescriptor(B)
    #    size=TupleTools.getindices(size(B), pB′),
    #    strides=TupleTools.getindices(strides(B), pB′))

    modeC, modeA, modeB = _indextuple_to_mode(pC, pA, pB)

    alignmentRequirementC = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), C, descC, alignmentRequirementC)
    alignmentRequirementA = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), A, descA, alignmentRequirementA)
    alignmentRequirementB = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), B, descB, alignmentRequirementB)

    desc = Ref{cutensorContractionDescriptor_t}()
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                                      descA, modeA, alignmentRequirementA[],
                                      descB, modeB, alignmentRequirementB[],
                                      descC, modeC, alignmentRequirementC[],
                                      descC, modeC, alignmentRequirementC[],
                                      cutensorComputeType(eltype(C)))

    return desc
end

function _indextuple_to_mode(pC::Index2Tuple, pA::Index2Tuple, pB::Index2Tuple)
    NoA = TO.numout(pA)
    NoB = TO.numin(pB)
    Nc = TO.numin(pA)

    modec = ntuple(n -> NoA + NoB + n, Nc)
    modeoA = ntuple(n -> n, NoA)
    modeoB = ntuple(n -> NoA + n, NoB)

    modeC = collect(Cint, linearize(pC))
    modeA = collect(Cint, (modeoA..., modec...))
    modeB = collect(Cint, (modec..., modeoB...))

    return modeC, modeA, modeB
end

function TO.tensortrace!(C::AnyCuArray, pC::Index2Tuple,
                         A::AnyCuArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number)
    return tensortrace!(C, pC, A, pA, conjA, α, β, StridedCUDA())
end
function TO.tensortrace!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::StridedCUDA)
    tensortrace!(StridedView(C), pC, StridedView(A), pA, conjA, α, β, backend)
    return C
end
function TO.tensortrace!(C::CuStridedView, pC::Index2Tuple,
                         A::CuStridedView, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, ::StridedCUDA=StridedCUDA())
    # TODO: check if these checks are necessary
    # -- in principle cuTENSOR already checks this, but the error messages are less clear
    TO.argcheck_tensortrace(C, pC, A, pA)
    TO.dimcheck_tensortrace(C, pC, A, pA)

    descC = CuTensorDescriptor(C)
    modeC = collect(Cint, 1:ndims(C))
    descA = CuTensorDescriptor(TO.flag2op(conjA)(A);
                               size=(size(C)..., TupleTools.getindices(size(A), pA[1])...),
                               strides=(TupleTools.getindices(strides(A), linearize(pC))...,
                                        (TupleTools.getindices(strides(A), pA[1]) .+
                                         TupleTools.getindices(strides(A), pA[2]))...))
    modeA = collect(Cint, 1:ndims(A))

    T = eltype(C)
    typeCompute = cutensorComputeType(T)
    function workspacesize()
        out = Ref{UInt64}(C_NULL)
        cutensorReductionGetWorkspace(handle(),
                                      A, descA, modeA,
                                      C, descC, modeC,
                                      C, descC, modeC,
                                      CUTENSOR_OP_ADD, typeCompute,
                                      out)
        return out[]
    end
    with_workspace(workspacesize, 1 << 13) do workspace
        return cutensorReduction(handle(),
                                 T[α], A, descA, modeA,
                                 T[β], C, descC, modeC,
                                 C, descC, modeC,
                                 CUTENSOR_OP_ADD, typeCompute,
                                 workspace, sizeof(workspace), default_stream())
    end
    return C
end

#-------------------------------------------------------------------------------------------
# Allocations
#-------------------------------------------------------------------------------------------

function TO.tensoradd_type(TC, pC::Index2Tuple, ::AnyCuArray, conjA::Symbol)
    return CuArray{TC,TO.numind(pC)}
end

function TO.tensorcontract_type(TC, pC::Index2Tuple,
                                ::AnyCuArray, pA::Index2Tuple, conjA::Symbol,
                                ::AnyCuArray, pB::Index2Tuple, conjB::Symbol)
    return CuArray{TC,TO.numind(pC)}
end

TO.tensorfree!(C::CuArray) = CUDA.unsafe_free!(C)

#-------------------------------------------------------------------------------------------
# Backend
#-------------------------------------------------------------------------------------------

const cuTENSORBackend = TO.Backend{:cuTENSOR}

function TO.tensoradd!(C::AbstractArray, pC::Index2Tuple,
                       A::AbstractArray, conjA::Symbol, α::Number, β::Number,
                       backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    tensoradd!(C_cuda, pC, A, conjA, α, β, backend)
    copyto!(C, collect(C_cuda))
    return C
end

function TO.tensoradd!(C::CuArray, pC::Index2Tuple,
                       A::AbstractArray, conjA::Symbol, α::Number, β::Number,
                       ::cuTENSORBackend)
    return tensoradd!(C, pC, adapt(CuArray, A), conjA, α, β)
end

function TO.tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                            A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    tensorcontract!(C_cuda, pC, A, pA, conjA, B, pB, conjB, α, β, backend)
    copyto!(C, collect(C_cuda))
    return C
end
function TO.tensorcontract!(C::CuArray, pC::Index2Tuple,
                            A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, ::cuTENSORBackend)
    return tensorcontract!(C, pC, adapt(CuArray, A), pA, conjA, adapt(CuArray, B), pB,
                           conjB, α, β)
end

function TO.tensortrace!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    tensortrace!(C_cuda, pC, A, pA, conjA, α, β, backend)
    copyto!(C, collect(C_cuda))
    return C
end
function TO.tensortrace!(C::CuArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, ::cuTENSORBackend)
    return tensortrace!(C, pC, adapt(CuArray, A), pA, conjA, α, β)
end

function TO.tensoradd_type(TC, pC::Index2Tuple, ::AbstractArray,
                           conjA::Symbol, ::cuTENSORBackend)
    return CuArray{TC,TO.numind(pC)}
end

function TO.tensorcontract_type(TC, pC::Index2Tuple,
                                ::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                                ::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                                ::cuTENSORBackend)
    return CuArray{TC,TO.numind(pC)}
end

function TO.tensoralloc_add(TC, pC, A::AbstractArray, conjA, istemp,
                            ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pC)}
    structure = TO.tensoradd_structure(pC, A, conjA)
    return tensoralloc(ttype, structure, istemp)::ttype
end

function TO.tensoralloc_contract(TC, pC,
                                 A::AbstractArray, pA, conjA,
                                 B::AbstractArray, pB, conjB,
                                 istemp, ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pC)}
    structure = TO.tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)
    return tensoralloc(ttype, structure, istemp)::ttype
end

TO.tensorfree!(C::CuArray, ::cuTENSORBackend) = tensorfree!(C)

end
