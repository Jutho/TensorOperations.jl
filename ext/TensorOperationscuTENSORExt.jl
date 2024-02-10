module TensorOperationscuTENSORExt

#=
in general, the cuTENSOR operations work as follows:
1. create a plan for the operation
    - make tensor descriptors for the input and output arrays
    - describe the operation to be performed:
        labels for permutations
        scalar factors
        unary operations (e.g. conjugation)
        binary reduction operations (e.g. addition)
        scalar compute type
2. execute the plan on given tensors
    - forward pointers to the input and output arrays
=#

using TensorOperations
using TensorOperations: TensorOperations as TO

using cuTENSOR
using cuTENSOR: CUDA
using cuTENSOR: CUTENSOR_OP_IDENTITY, CUTENSOR_OP_CONJ, CUTENSOR_OP_ADD

using CUDA: CuArray, AnyCuArray
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
const SUPPORTED_CUARRAYS = Union{AnyCuArray,CuStridedView}
const cuTENSORBackend = TO.Backend{:cuTENSOR}


function TO.tensorscalar(C::SUPPORTED_CUARRAYS)
    return ndims(C) == 0 ? tensorscalar(collect(C)) : throw(DimensionMismatch())
end

function tensorop(A::AnyCuArray, conjA::Symbol=:N)
    return (eltype(A) <: Real || conjA === :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
end
function tensorop(A::CuStridedView, conjA::Symbol=:N)
    return if (eltype(A) <: Real || !xor(conjA === :C, A.op === conj))
        CUTENSOR_OP_IDENTITY
    else
        CUTENSOR_OP_CONJ
    end
end

#-------------------------------------------------------------------------------------------
# Default backends
#-------------------------------------------------------------------------------------------

# making sure that if no backend is specified, the cuTENSOR backend is used:

function TO.tensoradd!(C::SUPPORTED_CUARRAYS, pC::Index2Tuple, A::SUPPORTED_CUARRAYS, conjA::Symbol,
                       α::Number, β::Number)
    return tensoradd!(C, pC, A, conjA, α, β, cuTENSORBackend())
end
function TO.tensorcontract!(C::SUPPORTED_CUARRAYS, pC::Index2Tuple,
                            A::SUPPORTED_CUARRAYS, pA::Index2Tuple, conjA::Symbol,
                            B::SUPPORTED_CUARRAYS, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, cuTENSORBackend())
end
function TO.tensortrace!(C::SUPPORTED_CUARRAYS, pC::Index2Tuple,
                         A::SUPPORTED_CUARRAYS, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number)
    return tensortrace!(C, pC, A, pA, conjA, α, β, cuTENSORBackend())
end

# making sure that if the backend is specified, arrays are converted to CuArrays

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
                            α::Number, β::Number, backend::cuTENSORBackend)
    return tensorcontract!(C, pC, adapt(CuArray, A), pA, conjA, adapt(CuArray, B), pB,
                           conjB, α, β, backend)
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
                         α::Number, β::Number, backend::cuTENSORBackend)
    return tensortrace!(C, pC, adapt(CuArray, A), pA, conjA, α, β, backend)
end

function TO.tensoralloc_add(TC, pC, A::AbstractArray, conjA, istemp,
                            ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pC)}
    structure = TO.tensoradd_structure(pC, A, conjA)
    return TO.tensoralloc(ttype, structure, istemp)::ttype
end

function TO.tensoralloc_contract(TC, pC,
                                 A::AbstractArray, pA, conjA,
                                 B::AbstractArray, pB, conjB,
                                 istemp, ::cuTENSORBackend)
    ttype = CuArray{TC,TO.numind(pC)}
    structure = TO.tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)
    return tensoralloc(ttype, structure, istemp)::ttype
end

#-------------------------------------------------------------------------------------------
# tensoradd!
#-------------------------------------------------------------------------------------------

function TO.tensoradd!(C::CuArray, pC::Index2Tuple,
                       A::CuArray, conjA::Symbol,
                       α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Cinds = collect.(TO.add_labels(pC))
    opA = tensorop(A, conjA)

    # dispatch to cuTENSOR
    return if iszero(β)
        cuTENSOR.permute!(α, A, Ainds, opA, C, Cinds)
    else
        cuTENSOR.elementwise_binary_execute!(α,
                                             A, Ainds, opA,
                                             β,
                                             C, Cinds, CUTENSOR_OP_IDENTITY,
                                             C, Cinds,
                                             CUTENSOR_OP_ADD)
    end
end

function TO.tensorcontract!(C::CuArray, pC::Index2Tuple,
                            A::CuArray, pA::Index2Tuple, conjA::Symbol,
                            B::CuArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Binds, Cinds = collect.(TO.contract_labels(pA, pB, pC))
    opA = tensorop(A, conjA)
    opB = tensorop(B, conjB)
    
    # dispatch to cuTENSOR
    return cuTENSOR.contract!(α,
                              A, Ainds, opA,
                              B, Binds, opB,
                              β,
                              C, Cinds, CUTENSOR_OP_IDENTITY,
                              CUTENSOR_OP_IDENTITY)
end

function TO.tensortrace!(C::CuArray, pC::Index2Tuple,
                         A::CuArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Cinds = collect.(TO.trace_labels(pC, pA...))
    opA = tensorop(A, conjA)

    # map to reduction operation
    plan = plan_trace(A, Ainds, opA, C, Cinds, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD)
    return cuTENSOR.reduce!(plan, α, A, β, C)
end

function plan_trace(@nospecialize(A::AbstractArray), Ainds::cuTENSOR.ModeType,
                    opA::cuTENSOR.cutensorOperator_t,
                    @nospecialize(C::AbstractArray), Cinds::cuTENSOR.ModeType,
                    opC::cuTENSOR.cutensorOperator_t,
                    opReduce::cuTENSOR.cutensorOperator_t;
                    jit::cuTENSOR.cutensorJitMode_t=cuTENSOR.JIT_MODE_NONE,
                    workspace::cuTENSOR.cutensorWorksizePreference_t=cuTENSOR.WORKSPACE_DEFAULT,
                    algo::cuTENSOR.cutensorAlgo_t=cuTENSOR.ALGO_DEFAULT,
                    compute_type::Union{DataType,cuTENSOR.cutensorComputeDescriptorEnum,
                                        Nothing}=nothing)
    !cuTENSOR.is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !cuTENSOR.is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !cuTENSOR.is_binary(opReduce) && throw(ArgumentError("opReduce must be a binary op!"))
    
    # TODO: check if this can be avoided, available in caller
    # TODO: cuTENSOR will allocate sizes and strides anyways, could use that here
    _, cindA1, cindA2 = TO.trace_indices(tuple(Ainds...), tuple(Cinds...))
    
    # add strides of cindA2 to strides of cindA1 -> selects diagonal
    stA = strides(A)
    for (i, j) in zip(cindA1, cindA2)
        stA = Base.setindex(stA, stA[i] + stA[j], i)
    end
    szA = TT.deleteat(size(A), cindA2)
    stA′ = TT.deleteat(stA, cindA2)
    
    descA = cuTENSOR.CuTensorDescriptor(A; size=szA, strides=stA′)
    descC = cuTENSOR.CuTensorDescriptor(C)
    
    modeA = collect(Cint, deleteat!(Ainds, cindA2))
    modeC = collect(Cint, Cinds)
    
    actual_compute_type = if compute_type === nothing
        cuTENSOR.reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end
    
    desc = Ref{cuTENSOR.cutensorOperationDescriptor_t}()
    cuTENSOR.cutensorCreateReduction(cuTENSOR.handle(),
                            desc,
                            descA, modeA, opA,
                            descC, modeC, opC,
                            descC, modeC, opReduce,
                            actual_compute_type)

    plan_pref = Ref{cuTENSOR.cutensorPlanPreference_t}()
    cuTENSOR.cutensorCreatePlanPreference(cuTENSOR.handle(), plan_pref, algo, jit)

    return cuTENSOR.CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

#-------------------------------------------------------------------------------------------
# Allocations
#-------------------------------------------------------------------------------------------

function TO.tensoradd_type(TC, pC::Index2Tuple, ::SUPPORTED_CUARRAYS, conjA::Symbol)
    return CUDA.CuArray{TC,TO.numind(pC)}
end

function TO.tensorcontract_type(TC, pC::Index2Tuple,
                                ::SUPPORTED_CUARRAYS, pA::Index2Tuple, conjA::Symbol,
                                ::SUPPORTED_CUARRAYS, pB::Index2Tuple, conjB::Symbol)
    return CUDA.CuArray{TC,TO.numind(pC)}
end

TO.tensorfree!(C::SUPPORTED_CUARRAYS) = CUDA.unsafe_free!(C)

#-------------------------------------------------------------------------------------------
# Backend
#-------------------------------------------------------------------------------------------

end
