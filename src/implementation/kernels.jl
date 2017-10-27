# implementation/kernels.jl
#
# Implements the microkernels for solving the subproblems of the various problems.

@generated function add_micro!(α, A::StridedData{N}, β, C::StridedData{N}, dims::IndexTuple{N}, offsetA::Int, offsetC::Int) where N
    quote
        startA = A.start+offsetA
        stridesA = A.strides
        startC = C.start+offsetC
        stridesC = C.strides
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(α,A[indA],β,C[indC]))
        return C
    end
end

@generated function trace_micro!(α, A::StridedData{N}, β, C::StridedData{N}, dims::IndexTuple{N}, offsetA::Int, offsetC::Int) where N
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

@generated function contract_micro!(α, A::StridedData{N}, B::StridedData{N}, β, C::StridedData{N}, dims::IndexTuple{N}, offsetA, offsetB, offsetC) where N
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
