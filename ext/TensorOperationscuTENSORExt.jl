module TensorOperationscuTENSORExt

using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: cuTENSORBackend

using cuTENSOR
using cuTENSOR: OP_IDENTITY, OP_CONJ, OP_ADD
using cuTENSOR: is_unary, is_binary
using cuTENSOR: handle, stream
using cuTENSOR: cutensorWorksizePreference_t, cutensorAlgo_t, cutensorOperationDescriptor_t,
                cutensorOperator_t, cutensorJitMode_t, cutensorPlanPreference_t,
                cutensorComputeDescriptorEnum
using cuTENSOR: WORKSPACE_DEFAULT, ALGO_DEFAULT, JIT_MODE_NONE
using cuTENSOR: cutensorCreatePlanPreference, cutensorPlan, CuTensorPlan,
                CuTensorDescriptor, ModeType

using cuTENSOR: elementwise_binary_execute!, permute!, contract!, reduce!

# reduce
using cuTENSOR: reduction_compute_types, cutensorCreateReduction

using cuTENSOR: CUDA
using CUDA: CuArray, StridedCuArray, DenseCuArray, AnyCuArray
# this might be dependency-piracy, but removes a dependency from the main package
using CUDA.Adapt: adapt

using Strided
using TupleTools: TupleTools as TT

StridedViewsCUDAExt = Base.get_extension(Strided.StridedViews, :StridedViewsCUDAExt)
isnothing(StridedViewsCUDAExt) && error("StridedViewsCUDAExt not found")

#-------------------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------------------

const CuStridedView = StridedViewsCUDAExt.CuStridedView
const SUPPORTED_CUARRAYS = (:StridedCuArray, :CuStridedView)

function TO.tensorscalar(C::StridedCuArray)
    return ndims(C) == 0 ? tensorscalar(collect(C)) : throw(DimensionMismatch())
end
function TO.tensorscalar(C::CuStridedView)
    return ndims(C) == 0 ? CUDA.@allowscalar(C[]) : throw(DimensionMismatch())
end

function tensorop(A::StridedCuArray, conjA::Bool=false)
    return (eltype(A) <: Real || !conjA) ? OP_IDENTITY : OP_CONJ
end
function tensorop(A::CuStridedView, conjA::Bool=false)
    return if (eltype(A) <: Real || !xor(conjA, A.op === conj))
        OP_IDENTITY
    else
        OP_CONJ
    end
end

#-------------------------------------------------------------------------------------------
# Default backends
#-------------------------------------------------------------------------------------------

# making sure that if no backend is specified, the cuTENSOR backend is used:

for ArrayType in SUPPORTED_CUARRAYS
    @eval function TO.tensoradd!(C::$ArrayType, A::$ArrayType, pA::Index2Tuple,
                                 conjA::Bool,
                                 α::Number, β::Number)
        return tensoradd!(C, A, pA, conjA, α, β, cuTENSORBackend())
    end
    @eval function TO.tensorcontract!(C::$ArrayType,
                                      A::$ArrayType, pA::Index2Tuple, conjA::Bool,
                                      B::$ArrayType, pB::Index2Tuple, conjB::Bool,
                                      pAB::Index2Tuple, α::Number, β::Number)
        return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, cuTENSORBackend())
    end
    @eval function TO.tensortrace!(C::$ArrayType,
                                   A::$ArrayType, p::Index2Tuple, q::Index2Tuple,
                                   conjA::Bool,
                                   α::Number, β::Number)
        return tensortrace!(C, A, p, q, conjA, α, β, cuTENSORBackend())
    end
    @eval function TO.tensoradd_type(TC, ::$ArrayType, pA::Index2Tuple, conjA::Bool)
        return CUDA.CuArray{TC,TO.numind(pA)}
    end
    @eval function TO.tensorcontract_type(TC,
                                          ::$ArrayType, pA::Index2Tuple, conjA::Bool,
                                          ::$ArrayType, pB::Index2Tuple, conjB::Bool,
                                          pAB::Index2Tuple)
        return CUDA.CuArray{TC,TO.numind(pAB)}
    end
    @eval TO.tensorfree!(C::$ArrayType) = TO.tensorfree!(C::$ArrayType, cuTENSORBackend())
end

# making sure that if the backend is specified, arrays are converted to CuArrays

function TO.tensoradd!(C::AbstractArray,
                       A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number,
                       backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    tensoradd!(C_cuda, A_cuda, pA, conjA, α, β, backend)
    C === C_cuda || copy!(C, C_cuda)
    return C
end
function TO.tensorcontract!(C::AbstractArray,
                            A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                            B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                            pAB::Index2Tuple,
                            α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    B_cuda = adapt(CuArray, B)
    tensorcontract!(C_cuda, A_cuda, pA, conjA, B_cuda, pB, conjB, pAB, α, β, backend)
    C === C_cuda || copy!(C, C_cuda)
    return C
end
function TO.tensortrace!(C::AbstractArray,
                         A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                         α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    tensortrace!(C_cuda, A_cuda, p, q, conjA, α, β, backend)
    C === C_cuda || copy!(C, C_cuda)
    return C
end

function TO.tensoralloc_add(TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                            istemp::Bool,
                            ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp)::ttype
end

function TO.tensoralloc_contract(TC,
                                 A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                                 B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                                 pAB::Index2Tuple,
                                 istemp::Bool, ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure, istemp)::ttype
end

function TO.tensorfree!(C::CuArray, ::cuTENSORBackend)
    CUDA.unsafe_free!(C)
    return nothing
end

# Convert all implementations to StridedViews
# This should work for wrapper types that are supported by StridedViews
function TO.tensoradd!(C::AnyCuArray,
                       A::AnyCuArray, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number, backend::cuTENSORBackend)
    tensoradd!(StridedView(C), StridedView(A), pA, conjA, α, β, backend)
    return C
end
function TO.tensorcontract!(C::AnyCuArray, A::AnyCuArray,
                            pA::Index2Tuple, conjA::Bool, B::AnyCuArray,
                            pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number,
                            β::Number,
                            backend::cuTENSORBackend)
    tensorcontract!(StridedView(C), StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB, pAB, α, β, backend)
    return C
end
function TO.tensortrace!(C::AnyCuArray,
                         A::AnyCuArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                         α::Number, β::Number, backend::cuTENSORBackend)
    tensortrace!(StridedView(C), StridedView(A), p, q, conjA, α, β, backend)
    return C
end

#-------------------------------------------------------------------------------------------
# Implementation
#-------------------------------------------------------------------------------------------

function TO.tensoradd!(C::CuStridedView,
                       A::CuStridedView, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Cinds = collect.(TO.add_labels(pA))
    opA = tensorop(A, conjA)

    # dispatch to cuTENSOR
    return if iszero(β)
        permute!(α, A, Ainds, opA, C, Cinds)
    else
        elementwise_binary_execute!(α,
                                    A, Ainds, opA,
                                    β,
                                    C, Cinds, OP_IDENTITY,
                                    C, Cinds,
                                    OP_ADD)
    end
end

function TO.tensorcontract!(C::CuStridedView,
                            A::CuStridedView, pA::Index2Tuple, conjA::Bool,
                            B::CuStridedView, pB::Index2Tuple, conjB::Bool,
                            pAB::Index2Tuple,
                            α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Binds, Cinds = collect.(TO.contract_labels(pA, pB, pAB))
    opA = tensorop(A, conjA)
    opB = tensorop(B, conjB)

    # dispatch to cuTENSOR
    return contract!(α,
                     A, Ainds, opA,
                     B, Binds, opB,
                     β,
                     C, Cinds, OP_IDENTITY,
                     OP_IDENTITY)
end

function TO.tensortrace!(C::CuStridedView,
                         A::CuStridedView, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                         α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Cinds = collect.(TO.trace_labels(p, q))
    opA = tensorop(A, conjA)

    # map to reduction operation
    plan = plan_trace(A, Ainds, opA, C, Cinds, OP_IDENTITY, OP_ADD)
    return reduce!(plan, α, A, β, C)
end

function cuTENSOR.CuTensorDescriptor(a::CuStridedView; size=size(a), strides=strides(a),
                                     eltype=eltype(a))
    sz = collect(Int64, size)
    st = collect(Int64, strides)
    alignment = UInt32(find_alignment(a))
    return cuTENSOR.CuTensorDescriptor(sz, st, eltype, alignment)
end

const MAX_ALIGNMENT = UInt(256) # This is the largest alignment of CUDA memory
"find the alignment of the first element of the view"
find_alignment(A::CuStridedView) = gcd(MAX_ALIGNMENT, convert(UInt, pointer(A)))

# trace!
# ------
# not actually part of cuTENSOR, just a special case of reduce

function plan_trace(@nospecialize(A::AbstractArray), Ainds::ModeType,
                    opA::cutensorOperator_t,
                    @nospecialize(C::AbstractArray), Cinds::ModeType,
                    opC::cutensorOperator_t,
                    opReduce::cutensorOperator_t;
                    jit::cutensorJitMode_t=JIT_MODE_NONE,
                    workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                    algo::cutensorAlgo_t=ALGO_DEFAULT,
                    compute_type::Union{DataType,cutensorComputeDescriptorEnum,
                                        Nothing}=nothing)
    !is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce) && throw(ArgumentError("opReduce must be a binary op!"))

    # TODO: check if this can be avoided, available in caller
    # TODO: cuTENSOR will allocate sizes and strides anyways, could use that here
    p, q = TO.trace_indices(tuple(Ainds...), tuple(Cinds...))

    # add strides of cindA2 to strides of cindA1 -> selects diagonal
    stA = strides(A)
    for (i, j) in zip(q...)
        stA = Base.setindex(stA, stA[i] + stA[j], i)
    end
    szA = TT.deleteat(size(A), q[2])
    stA′ = TT.deleteat(stA, q[2])

    descA = CuTensorDescriptor(A; size=szA, strides=stA′)
    descC = CuTensorDescriptor(C)

    modeA = collect(Cint, deleteat!(Ainds, q[2]))
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateReduction(handle(),
                            desc,
                            descA, modeA, opA,
                            descC, modeC, opC,
                            descC, modeC, opReduce,
                            actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    return CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

end
