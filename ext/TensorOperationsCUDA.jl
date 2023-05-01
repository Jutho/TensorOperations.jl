module TensorOperationsCUDA

using TensorOperations
using TensorOperations: AbstractBackend, Index2Tuple, IndexTuple, linearize, IndexError
using TupleTools

if isdefined(Base, :get_extension)
    using CUDA: CUDA, CuArray
    using CUDA.CUBLAS: CublasFloat, CublasReal
    using cuTENSOR: handle, CuTensorDescriptor, cudaDataType_t,
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
                    cutensorContraction
else
    using ..CUDA: CUDA, CuArray
    using ..CUDA.CUBLAS: CublasFloat, CublasReal
    using ..cuTENSOR: handle, CuTensorDescriptor, cudaDataType_t,
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
                      cutensorContraction
end

if isdefined(CUDA, :default_stream)
    const default_stream = CUDA.default_stream
else
    const default_stream = CUDA.CuDefaultStream
end

if isdefined(CUDA, :with_workspace)
    # for CUDA 3.3 and later
    using CUDA: with_workspace
else
    # for CUDA 3.2 and earlier
    # this is a minimum wrapper for @workspace that supports the interface of
    # with_workspace used below. It is "hidden" behind @eval so the fact that
    # @workspace is not defined on CUDA 3.3+ does not throw an error
    @eval @inline function with_workspace(f, size, fallback)
        CUDA.@workspace size = size() fallback = fallback workspace -> f(workspace)
    end
end

function TensorOperations.tensorscalar(C::CuArray)
    return ndims(C) == 0 ? collect(C)[] : throw(DimensionMismatch())
end

# ---------------------------------------------------------------------------------------- #
# tensoradd!
# ---------------------------------------------------------------------------------------- #

function TensorOperations.tensoradd!(::AbstractBackend, C::CuArray, A::CuArray,
                                     pA::Index2Tuple,
                                     conjA::Symbol, α::Number, β::Number)
    N = ndims(C)
    N == ndims(A) || throw(DimensionMismatch("ndims(A) ≠ ndims(C)"))
    N == length(pA[1]) + length(pA[2]) ||
        throw(IndexError("Invalid permutation of length $N: $pA"))

    T = eltype(C)

    conjA == :N || conjA == :C ||
        throw(ArgumentError("Value of conjA should be :N or :C instead of $conjA"))
    opA = (T <: Real || conjA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opAC = CUTENSOR_OP_ADD

    descA = CuTensorDescriptor(A; op=opA)
    descC = CuTensorDescriptor(C; op=CUTENSOR_OP_IDENTITY)
    # descD = descC
    typeCompute = convert(cudaDataType_t, T)
    modeA = collect(Cint, 1:N)
    modeC = collect(Cint, linearize(pA))
    stream = default_stream()
    if β == zero(β)
        cutensorPermutation(handle(), T[α], A, descA, modeA, C, descC, modeC,
                            typeCompute, stream)
    else
        cutensorElementwiseBinary(handle(), T[α], A, descA, modeA, T[β], C, descC,
                                  modeC, C, descC, modeC, opAC, typeCompute, stream)
    end

    return C
end

# ---------------------------------------------------------------------------------------- #
# tensorcontract!
# ---------------------------------------------------------------------------------------- #

function TensorOperations.tensorcontract!(::AbstractBackend, C::CuArray, pC::Index2Tuple,
                                          A::CuArray, pA::Index2Tuple, conjA::Symbol,
                                          B::CuArray, pB::Index2Tuple, conjB::Symbol,
                                          α, β)
    (length(pA[1]) + length(pA[2]) == ndims(A) && TupleTools.isperm(linearize(pA))) ||
        throw(IndexError("invalid permutation of A of length $(ndims(A)): $pA"))
    (length(pB[1]) + length(pB[2]) == ndims(B) && TupleTools.isperm(linearize(pB))) ||
        throw(IndexError("invalid permutation of B of length $(ndims(B)): $pB"))
    (length(pA[1]) + length(pB[2]) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (length(pC[1]) + length(pC[2]) == ndims(C) && TupleTools.isperm(linearize(pC))) ||
        throw(IndexError("invalid permutation of C of length $(ndims(C)): $pC"))

    sizeA = i -> size(A, i)
    sizeB = i -> size(B, i)
    # sizeC = i -> size(C, i)

    csizeA = sizeA.(pA[2])
    csizeB = sizeB.(pB[1])
    osizeA = sizeA.(pA[1])
    osizeB = sizeB.(pB[2])

    csizeA == csizeB ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    sizeAB = let osize = (osizeA..., osizeB...)
        i -> osize[i]
    end
    sizeAB.(linearize(pC)) == size(C) ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    TC = eltype(C)

    conjA == :N || conjA == :C ||
        throw(ArgumentError("Value of conjA should be :N or :C instead of $conjA"))
    conjB == :N || conjB == :C ||
        throw(ArgumentError("Value of conjB should be :N or :C instead of $conjB"))

    opA = (TC <: Real || conjA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opB = (TC <: Real || conjB == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    # opC = CUTENSOR_OP_IDENTITY

    strideA = i -> stride(A, i)
    strideB = i -> stride(B, i)

    cstrideA = strideA.(pA[2])
    cstrideB = strideB.(pB[1])
    ostrideA = strideA.(pA[1])
    ostrideB = strideB.(pB[2])

    descA = CuTensorDescriptor(A; op=opA, size=(osizeA..., csizeA...),
                               strides=(ostrideA..., cstrideA...))
    descB = CuTensorDescriptor(B; op=opB, size=(osizeB..., csizeB...),
                               strides=(ostrideB..., cstrideB...))
    descC = CuTensorDescriptor(C)
    T = eltype(C)
    typeCompute = cutensorComputeType(T)
    # opOut = CUTENSOR_OP_IDENTITY

    NoA = length(pA[1])
    NoB = length(pB[2])
    Nc = length(pA[2])
    modeoA = ntuple(n -> n, NoA)
    modeoB = ntuple(n -> NoA + n, NoB)
    modec = ntuple(n -> NoA + NoB + n, Nc)

    modeA = collect(Cint, (modeoA..., modec...))
    modeB = collect(Cint, (modeoB..., modec...))
    modeC = collect(Cint, linearize(pC))

    algo = CUTENSOR_ALGO_DEFAULT
    stream = default_stream()
    pref = CUTENSOR_WORKSPACE_RECOMMENDED

    alignmentRequirementA = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), A, descA, alignmentRequirementA)
    alignmentRequirementB = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), B, descB, alignmentRequirementB)
    alignmentRequirementC = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), C, descC, alignmentRequirementC)
    desc = Ref{cutensorContractionDescriptor_t}()
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                                      descA, modeA, alignmentRequirementA[],
                                      descB, modeB, alignmentRequirementB[],
                                      descC, modeC, alignmentRequirementC[],
                                      descC, modeC, alignmentRequirementC[],
                                      typeCompute)

    find = Ref{cutensorContractionFind_t}()
    cutensorInitContractionFind(handle(), find, algo)

    function workspacesize()
        out = Ref{UInt64}(C_NULL)
        cutensorContractionGetWorkspace(handle(), desc, find, pref, out)
        return out[]
    end
    with_workspace(workspacesize, 1 << 27) do workspace
        plan_ref = Ref{cutensorContractionPlan_t}()
        cutensorInitContractionPlan(handle(), plan_ref, desc, find, sizeof(workspace))

        return cutensorContraction(handle(), plan_ref, T[α], A, B, T[β], C, C,
                                   workspace, sizeof(workspace), stream)
    end

    return C
end

# ---------------------------------------------------------------------------------------- #
# tensortrace!
# ---------------------------------------------------------------------------------------- #

function TensorOperations.tensortrace!(::AbstractBackend, C::CuArray, pC::Index2Tuple,
                                       A::CuArray, pA::Index2Tuple, conjA::Symbol, α, β)
    T = eltype(C)
    NA, NC = ndims(A), ndims(C)
    NC == length(linearize(pC)) ||
        throw(IndexError("invalid selection of $NC out of $NA: $pC"))
    NA - NC == 2 * length(pA[1]) == 2 * length(pA[2]) ||
        throw(IndexError("invalid number of trace dimensions"))

    opA = (T <: Real || conjA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opReduce = CUTENSOR_OP_ADD

    sizeA = i -> size(A, i)
    strideA = i -> stride(A, i)
    tracesize = sizeA.(pA[1])
    tracesize == sizeA.(pA[2]) || throw(DimensionMismatch("non-matching trace sizes"))
    size(C) == sizeA.(linearize(pC)) || throw(DimensionMismatch("non-matching sizes"))

    newstrides = (strideA.(linearize(pC))..., (strideA.(pA[1]) .+ strideA.(pA[2]))...)
    newsize = (size(C)..., tracesize...)
    descA = CuTensorDescriptor(A; op=opA, size=newsize, strides=newstrides)
    descC = CuTensorDescriptor(C; op=CUTENSOR_OP_IDENTITY)
    # descD = descC
    typeCompute = cutensorComputeType(T)
    modeA = collect(Cint, 1:NA)
    modeC = collect(Cint, 1:NC)
    stream = default_stream()
    function workspacesize()
        out = Ref{UInt64}(C_NULL)
        cutensorReductionGetWorkspace(handle(),
                                      A, descA, modeA,
                                      C, descC, modeC,
                                      C, descC, modeC,
                                      opReduce, typeCompute,
                                      out)
        return out[]
    end
    with_workspace(workspacesize, 1 << 13) do workspace
        return cutensorReduction(handle(),
                                 T[α], A, descA, modeA,
                                 T[β], C, descC, modeC,
                                 C, descC, modeC,
                                 opReduce, typeCompute,
                                 workspace, sizeof(workspace), stream)
    end
    return C
end

# ---------------------------------------------------------------------------------------- #
# JuliaAllocator
# ---------------------------------------------------------------------------------------- #

function TensorOperations.tensoradd_type(TC, A::CuArray, pA::Index2Tuple, conjA::Symbol)
    return CuArray{TC,sum(length.(pA))}
end

function TensorOperations.tensorcontract_type(TC, pC, A::CuArray, pA, conjA,
                                              B::CuArray, pB, conjB)
    return CuArray{TC,sum(length.(pC))}
end

end
