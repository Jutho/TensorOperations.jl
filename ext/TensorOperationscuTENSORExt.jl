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
const SUPPORTED_CUARRAYS = (:AnyCuArray, :CuStridedView)
const cuTENSORBackend = TO.Backend{:cuTENSOR}

function TO.tensorscalar(C::AnyCuArray)
    return ndims(C) == 0 ? tensorscalar(collect(C)) : throw(DimensionMismatch())
end
function TO.tensorscalar(C::CuStridedView)
    return ndims(C) == 0 ? CUDA.@allowscalar(C[]) : throw(DimensionMismatch())
end

function tensorop(A::AnyCuArray, conjA::Symbol=:N)
    return (eltype(A) <: Real || conjA === :N) ? cuTENSOR.OP_IDENTITY : cuTENSOR.OP_CONJ
end
function tensorop(A::CuStridedView, conjA::Symbol=:N)
    return if (eltype(A) <: Real || !xor(conjA === :C, A.op === conj))
        cuTENSOR.OP_IDENTITY
    else
        cuTENSOR.OP_CONJ
    end
end

#-------------------------------------------------------------------------------------------
# Default backends
#-------------------------------------------------------------------------------------------

# making sure that if no backend is specified, the cuTENSOR backend is used:

for ArrayType in SUPPORTED_CUARRAYS
    @eval function TO.tensoradd!(C::$ArrayType, pC::Index2Tuple, A::$ArrayType,
                                 conjA::Symbol,
                                 α::Number, β::Number)
        return tensoradd!(C, pC, A, conjA, α, β, cuTENSORBackend())
    end
    @eval function TO.tensorcontract!(C::$ArrayType, pC::Index2Tuple,
                                      A::$ArrayType, pA::Index2Tuple, conjA::Symbol,
                                      B::$ArrayType, pB::Index2Tuple, conjB::Symbol,
                                      α::Number, β::Number)
        return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, cuTENSORBackend())
    end
    @eval function TO.tensortrace!(C::$ArrayType, pC::Index2Tuple,
                                   A::$ArrayType, pA::Index2Tuple, conjA::Symbol,
                                   α::Number, β::Number)
        return tensortrace!(C, pC, A, pA, conjA, α, β, cuTENSORBackend())
    end

    @eval function TO.tensoradd_type(TC, pC::Index2Tuple, ::$ArrayType, conjA::Symbol)
        return CUDA.CuArray{TC,TO.numind(pC)}
    end

    @eval function TO.tensorcontract_type(TC, pC::Index2Tuple,
                                          ::$ArrayType, pA::Index2Tuple, conjA::Symbol,
                                          ::$ArrayType, pB::Index2Tuple, conjB::Symbol)
        return CUDA.CuArray{TC,TO.numind(pC)}
    end

    @eval TO.tensorfree!(C::$ArrayType) = TO.tensorfree!(C::$ArrayType, cuTENSORBackend())
end

# making sure that if the backend is specified, arrays are converted to CuArrays

function TO.tensoradd!(C::AbstractArray, pC::Index2Tuple,
                       A::AbstractArray, conjA::Symbol, α::Number, β::Number,
                       backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    tensoradd!(C_cuda, pC, A_cuda, conjA, α, β, backend)
    copyto!(C, C_cuda)
    return C
end
function TO.tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                            A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    B_cuda = adapt(CuArray, B)
    tensorcontract!(C_cuda, pC, A_cuda, pA, conjA, B_cuda, pB, conjB, α, β, backend)
    copyto!(C, C_cuda)
    return C
end
function TO.tensortrace!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::cuTENSORBackend)
    C_cuda = adapt(CuArray, C)
    A_cuda = adapt(CuArray, A)
    tensortrace!(C_cuda, pC, A_cuda, pA, conjA, α, β, backend)
    copyto!(C, C_cuda)
    return C
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

function TO.tensorfree!(C::CuStridedView, backend::cuTENSORBackend)
    CUDA.unsafe_free!(parent(C))
    return nothing
end

# Convert all implementations to StridedViews
function TO.tensoradd!(C::AnyCuArray, pC::Index2Tuple,
                       A::AnyCuArray, conjA::Symbol,
                       α::Number, β::Number, backend::cuTENSORBackend)
    tensoradd!(StridedView(C), pC, StridedView(A), conjA, α, β, backend)
    return C
end
function TO.tensorcontract!(C::AnyCuArray, pC::Index2Tuple, A::AnyCuArray,
                            pA::Index2Tuple, conjA::Symbol, B::AnyCuArray,
                            pB::Index2Tuple, conjB::Symbol, α::Number, β::Number,
                            backend::cuTENSORBackend)
    tensorcontract!(StridedView(C), pC, StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB, α, β, backend)
    return C
end
function TO.tensortrace!(C::AnyCuArray, pC::Index2Tuple,
                         A::AnyCuArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::cuTENSORBackend)
    tensortrace!(StridedView(C), pC, StridedView(A), pA, conjA, α, β, backend)
    return C
end
function TO.tensorfree!(C::AnyCuArray, backend::cuTENSORBackend)
    return tensorfree!(StridedView(C), backend)
end

#-------------------------------------------------------------------------------------------
# Implementation
#-------------------------------------------------------------------------------------------

function TO.tensoradd!(C::CuStridedView, pC::Index2Tuple,
                       A::CuStridedView, conjA::Symbol,
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
                                             C, Cinds, cuTENSOR.OP_IDENTITY,
                                             C, Cinds,
                                             cuTENSOR.OP_ADD)
    end
end

function TO.tensorcontract!(C::CuStridedView, pC::Index2Tuple,
                            A::CuStridedView, pA::Index2Tuple, conjA::Symbol,
                            B::CuStridedView, pB::Index2Tuple, conjB::Symbol,
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
                              C, Cinds, cuTENSOR.OP_IDENTITY,
                              cuTENSOR.OP_IDENTITY)
end

function TO.tensortrace!(C::CuStridedView, pC::Index2Tuple,
                         A::CuStridedView, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, ::cuTENSORBackend)
    # convert arguments
    Ainds, Cinds = collect.(TO.trace_labels(pC, pA...))
    opA = tensorop(A, conjA)

    # map to reduction operation
    plan = plan_trace(A, Ainds, opA, C, Cinds, cuTENSOR.OP_IDENTITY, cuTENSOR.OP_ADD)
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

    descA = CuTensorDescriptor(A; size=szA, strides=stA′)
    descC = CuTensorDescriptor(C)

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
# Implementations for StridedViews
#-------------------------------------------------------------------------------------------

# cuTENSOR does not readily support subarrays/views because they need to be strided, but
# StridedViews should always work. The following is a lot of code duplication from
# cuTENSOR.jl, but for now this will have to do.

using cuTENSOR: cutensorWorksizePreference_t, cutensorAlgo_t, cutensorComputeDescriptorEnum,
                CuTensorPlan, ModeType, cutensorOperator_t, cutensorJitMode_t,
                WORKSPACE_DEFAULT, ALGO_DEFAULT, JIT_MODE_NONE, CuTensorDescriptor,
                is_unary, is_binary, cutensorOperationDescriptor_t,
                cutensorCreateContraction,
                cutensorCreatePermutation, cutensorReduce, cutensorPlanPreference_t,
                plan_contraction, cutensorCreatePlanPreference, cutensorPermute,
                cutensorElementwiseBinaryExecute, cutensorContract,
                cutensorCreateElementwiseBinary, cutensorElementwiseBinaryExecute

function cuTENSOR.elementwise_binary_execute!(@nospecialize(alpha::Number),
                                              @nospecialize(A::CuStridedView),
                                              Ainds::ModeType,
                                              opA::cutensorOperator_t,
                                              @nospecialize(gamma::Number),
                                              @nospecialize(C::CuStridedView),
                                              Cinds::ModeType,
                                              opC::cutensorOperator_t,
                                              @nospecialize(D::CuStridedView),
                                              Dinds::ModeType,
                                              opAC::cutensorOperator_t;
                                              workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                                              algo::cutensorAlgo_t=ALGO_DEFAULT,
                                              compute_type::Union{DataType,
                                                                  cutensorComputeDescriptorEnum,
                                                                  Nothing}=nothing,
                                              plan::Union{CuTensorPlan,Nothing}=nothing)
    actual_plan = if plan === nothing
        cuTENSOR.plan_elementwise_binary(A, Ainds, opA,
                                         C, Cinds, opC,
                                         D, Dinds, opAC;
                                         workspace, algo, compute_type)
    else
        plan
    end

    cuTENSOR.elementwise_binary_execute!(actual_plan, alpha, A, gamma, C, D)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return D
end

function cuTENSOR.elementwise_binary_execute!(plan::CuTensorPlan,
                                              @nospecialize(alpha::Number),
                                              @nospecialize(A::CuStridedView),
                                              @nospecialize(gamma::Number),
                                              @nospecialize(C::CuStridedView),
                                              @nospecialize(D::CuStridedView))
    scalar_type = plan.scalar_type
    cutensorElementwiseBinaryExecute(cuTENSOR.handle(), plan,
                                     Ref{scalar_type}(alpha), A,
                                     Ref{scalar_type}(gamma), C, D,
                                     cuTENSOR.stream())
    return D
end

function cuTENSOR.plan_elementwise_binary(@nospecialize(A::CuStridedView), Ainds::ModeType,
                                          opA::cutensorOperator_t,
                                          @nospecialize(C::CuStridedView), Cinds::ModeType,
                                          opC::cutensorOperator_t,
                                          @nospecialize(D::CuStridedView), Dinds::ModeType,
                                          opAC::cutensorOperator_t;
                                          jit::cutensorJitMode_t=JIT_MODE_NONE,
                                          workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                                          algo::cutensorAlgo_t=ALGO_DEFAULT,
                                          compute_type::Union{DataType,
                                                              cutensorComputeDescriptorEnum,
                                                              Nothing}=eltype(C))
    !is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC) && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC

    actual_compute_type = if compute_type === nothing
        cuTENSOR.elementwise_binary_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateElementwiseBinary(cuTENSOR.handle(),
                                    desc,
                                    descA, modeA, opA,
                                    descC, modeC, opC,
                                    descD, modeD,
                                    opAC,
                                    actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(cuTENSOR.handle(), plan_pref, algo, jit)

    return CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function cuTENSOR.permute!(@nospecialize(alpha::Number),
                           @nospecialize(A::CuStridedView), Ainds::ModeType,
                           opA::cutensorOperator_t,
                           @nospecialize(B::CuStridedView), Binds::ModeType;
                           workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                           algo::cutensorAlgo_t=ALGO_DEFAULT,
                           compute_type::Union{DataType,cutensorComputeDescriptorEnum,
                                               Nothing}=nothing,
                           plan::Union{CuTensorPlan,Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_permutation(A, Ainds, opA,
                         B, Binds;
                         workspace, algo, compute_type)
    else
        plan
    end

    cuTENSOR.permute!(actual_plan, alpha, A, B)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return B
end

function cuTENSOR.permute!(plan::CuTensorPlan,
                           @nospecialize(alpha::Number),
                           @nospecialize(A::CuStridedView),
                           @nospecialize(B::CuStridedView))
    scalar_type = plan.scalar_type
    cutensorPermute(cuTENSOR.handle(), plan,
                    Ref{scalar_type}(alpha), A, B,
                    cuTENSOR.stream())
    return B
end

function plan_permutation(@nospecialize(A::CuStridedView), Ainds::ModeType,
                          opA::cutensorOperator_t,
                          @nospecialize(B::CuStridedView), Binds::ModeType;
                          jit::cutensorJitMode_t=JIT_MODE_NONE,
                          workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                          algo::cutensorAlgo_t=ALGO_DEFAULT,
                          compute_type::Union{DataType,cutensorComputeDescriptorEnum,
                                              Nothing}=nothing)
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)

    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)

    actual_compute_type = if compute_type === nothing
        cuTENSOR.permutation_compute_types[(eltype(A), eltype(B))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreatePermutation(cuTENSOR.handle(), desc,
                              descA, modeA, opA,
                              descB, modeB,
                              actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(cuTENSOR.handle(), plan_pref, algo, jit)

    return CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function cuTENSOR.contract!(@nospecialize(alpha::Number),
                            @nospecialize(A::CuStridedView), Ainds::ModeType,
                            opA::cutensorOperator_t,
                            @nospecialize(B::CuStridedView), Binds::ModeType,
                            opB::cutensorOperator_t,
                            @nospecialize(beta::Number),
                            @nospecialize(C::CuStridedView), Cinds::ModeType,
                            opC::cutensorOperator_t,
                            opOut::cutensorOperator_t;
                            jit::cutensorJitMode_t=JIT_MODE_NONE,
                            workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                            algo::cutensorAlgo_t=ALGO_DEFAULT,
                            compute_type::Union{DataType,cutensorComputeDescriptorEnum,
                                                Nothing}=nothing,
                            plan::Union{CuTensorPlan,Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut;
                         jit, workspace, algo, compute_type)
    else
        plan
    end

    cuTENSOR.contract!(actual_plan, alpha, A, B, beta, C)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function cuTENSOR.contract!(plan::CuTensorPlan,
                            @nospecialize(alpha::Number),
                            @nospecialize(A::CuStridedView),
                            @nospecialize(B::CuStridedView),
                            @nospecialize(beta::Number),
                            @nospecialize(C::CuStridedView))
    scalar_type = plan.scalar_type
    cutensorContract(cuTENSOR.handle(), plan,
                     Ref{scalar_type}(alpha), A, B,
                     Ref{scalar_type}(beta), C, C,
                     plan.workspace, sizeof(plan.workspace), cuTENSOR.stream())
    return C
end

function cuTENSOR.plan_contraction(@nospecialize(A::CuStridedView), Ainds::ModeType,
                                   opA::cutensorOperator_t,
                                   @nospecialize(B::CuStridedView), Binds::ModeType,
                                   opB::cutensorOperator_t,
                                   @nospecialize(C::CuStridedView), Cinds::ModeType,
                                   opC::cutensorOperator_t,
                                   opOut::cutensorOperator_t;
                                   jit::cutensorJitMode_t=JIT_MODE_NONE,
                                   workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                                   algo::cutensorAlgo_t=ALGO_DEFAULT,
                                   compute_type::Union{DataType,
                                                       cutensorComputeDescriptorEnum,
                                                       Nothing}=nothing)
    !is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB) && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut) && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        cuTENSOR.contraction_compute_types[(eltype(A), eltype(B), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateContraction(cuTENSOR.handle(),
                              desc,
                              descA, modeA, opA,
                              descB, modeB, opB,
                              descC, modeC, opC,
                              descC, modeC,
                              actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(cuTENSOR.handle(), plan_pref, algo, jit)

    return CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function cuTENSOR.reduce!(@nospecialize(alpha::Number),
                          @nospecialize(A::CuStridedView), Ainds::ModeType,
                          opA::cutensorOperator_t,
                          @nospecialize(beta::Number),
                          @nospecialize(C::CuStridedView), Cinds::ModeType,
                          opC::cutensorOperator_t,
                          opReduce::cutensorOperator_t;
                          workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
                          algo::cutensorAlgo_t=ALGO_DEFAULT,
                          compute_type::Union{DataType,cutensorComputeDescriptorEnum,
                                              Nothing}=nothing,
                          plan::Union{CuTensorPlan,Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_reduction(A, Ainds, opA, C, Cinds, opC, opReduce; workspace, algo,
                       compute_type)
    else
        plan
    end

    cuTENSOR.reduce!(actual_plan, alpha, A, beta, C)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function cuTENSOR.reduce!(plan::CuTensorPlan,
                          @nospecialize(alpha::Number),
                          @nospecialize(A::CuStridedView),
                          @nospecialize(beta::Number),
                          @nospecialize(C::CuStridedView))
    scalar_type = plan.scalar_type
    cuTENSOR.cutensorReduce(cuTENSOR.handle(), plan,
                            Ref{scalar_type}(alpha), A,
                            Ref{scalar_type}(beta), C, C,
                            plan.workspace, sizeof(plan.workspace), cuTENSOR.stream())
    return C
end

function cuTENSOR.plan_reduction(@nospecialize(A::CuStridedView), Ainds::ModeType,
                                 opA::cutensorOperator_t,
                                 @nospecialize(C::CuStridedView), Cinds::ModeType,
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
    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        cuTENSOR.reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateReduction(cuTENSOR.handle(),
                            desc,
                            descA, modeA, opA,
                            descC, modeC, opC,
                            descC, modeC, opReduce,
                            actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(cuTENSOR.handle(), plan_pref, algo, jit)

    return CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

end
