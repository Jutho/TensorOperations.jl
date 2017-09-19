# implementation/recursive.jl
#
# Implements a recursive divide and conquer approach to divide the problem into
# smaller subproblems by dividing those dimensions that lead to the largest
# jumps in memory. When the total length of the subproblem fits into BASELENGTH,
# the subproblem is handled by the kernels.

const BASELENGTH=2048

function add_rec!(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int}) where N
    if 2*prod(dims) <= BASELENGTH
        add_micro!(α, A, β, C, dims, offsetA, offsetC)
    else
        dmax = _indmax(_memjumps(dims, minstrides))
        newdim = dims[dmax] >> 1
        newdims = setindex(dims, newdim, dmax)
        add_rec!(α, A, β, C, newdims, offsetA, offsetC, minstrides)
        offsetA += newdim*A.strides[dmax]
        offsetC += newdim*C.strides[dmax]
        newdim = dims[dmax] - newdim
        newdims = setindex(dims, newdim, dmax)
        add_rec!(α, A, β, C, newdims, offsetA, offsetC, minstrides)
    end
    return C
end

function trace_rec!(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int}) where N
    if prod(dims) + prod(_filterdims(dims,C)) <= BASELENGTH
        trace_micro!(α, A, β, C, dims, offsetA, offsetC)
    else
        dmax = _indmax(_memjumps(dims, minstrides))
        newdim = dims[dmax] >> 1
        newdims = setindex(dims, newdim, dmax)
        trace_rec!(α, A, β, C, newdims, offsetA, offsetC, minstrides)
        offsetA += newdim*A.strides[dmax]
        offsetC += newdim*C.strides[dmax]
        newdim = dims[dmax] - newdim
        newdims = setindex(dims, newdim, dmax)
        if C.strides[dmax] == 0
            trace_rec!(α, A, _one, C, newdims, offsetA, offsetC, minstrides)
        else
            trace_rec!(α, A, β, C, newdims, offsetA, offsetC, minstrides)
        end
    end
    return C
end

function contract_rec!(α, A::StridedData{N}, B::StridedData{N}, β, C::StridedData{N},
    dims::NTuple{N, Int}, offsetA::Int, offsetB::Int, offsetC::Int, minstrides::NTuple{N, Int}) where N

    odimsA = _filterdims(_filterdims(dims, A), C)
    odimsB = _filterdims(_filterdims(dims, B), C)
    cdims = _filterdims(_filterdims(dims, A), B)
    oAlength = prod(odimsA)
    oBlength = prod(odimsB)
    clength = prod(cdims)

    if oAlength*oBlength + clength*(oAlength+oBlength) <= BASELENGTH
        contract_micro!(α, A, B, β, C, dims, offsetA, offsetB, offsetC)
    else
        if clength > oAlength && clength > oBlength
            dmax = _indmax(_memjumps(cdims, minstrides))
        elseif oAlength > oBlength
            dmax = _indmax(_memjumps(odimsA, minstrides))
        else
            dmax = _indmax(_memjumps(odimsB, minstrides))
        end
        newdim = dims[dmax] >> 1
        newdims = setindex(dims, newdim, dmax)
        contract_rec!(α, A, B, β, C, newdims, offsetA, offsetB, offsetC, minstrides)
        offsetA += newdim*A.strides[dmax]
        offsetB += newdim*B.strides[dmax]
        offsetC += newdim*C.strides[dmax]
        newdim = dims[dmax] - newdim
        newdims = setindex(dims, newdim, dmax)
        if C.strides[dmax] == 0 # dmax is contraction dimension: β -> 1
            contract_rec!(α, A, B, _one, C, newdims, offsetA, offsetB, offsetC, minstrides)
        else
            contract_rec!(α, A, B, β, C, newdims, offsetA, offsetB, offsetC, minstrides)
        end
    end
    return C
end
