# implementation/kernels.jl
#
# Implements the microkernels for solving the subproblems of the various problems.

@generated function add_micro!{N}(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int)
    quote
        startA = A.start+offsetA
        stridesA = A.strides
        startC = C.start+offsetC
        stridesC = C.strides
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(α,A[indA],β,C[indC]))
        return C
    end
end

@generated function trace_micro!{N}(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int)
    quote
        _scale!(C, β, dims, offsetC)
        startA = A.start+offsetA
        stridesA = A.strides
        startC = C.start+offsetC
        stridesC = C.strides
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(α,A[indA],_one,C[indC]))
        return C
    end
end

@generated function contract_micro!{N}(α, A::StridedData{N}, B::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA, offsetB, offsetC)
    quote
        _scale!(C, β, dims, offsetC)
        startA = A.start+offsetA
        stridesA = A.strides
        startB = B.start+offsetB
        stridesB = B.strides
        startC = C.start+offsetC
        stridesC = C.strides

        @stridedloops($N, dims, indA, startA, stridesA, indB, startB, stridesB, indC, startC, stridesC, @inbounds C[indC] = axpby(α, A[indA]*B[indB], _one, C[indC]))
        return C
    end
end
