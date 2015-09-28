# implementation/recursive.jl
#
# Implements a recursive divide and conquer approach to divide the problem into
# smaller subproblems by dividing those dimensions that lead to the largest
# jumps in memory. When the total length of the subproblem fits into BASELENGTH,
# the subproblem is handled by the kernels.

const BASELENGTH=2048

@generated function add_rec!{N}(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int})
    quote
        if 2*prod(dims) <= BASELENGTH
            add_micro!(α, A, β, C, dims, offsetA, offsetC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            @dividebody $N dmax dims offsetA A offsetC C begin
                add_rec!(α, A, β, C, dims, offsetA, offsetC, minstrides)
            end begin
                add_rec!(α, A, β, C, dims, offsetA, offsetC, minstrides)
            end
        end
        return C
    end
end

@generated function trace_rec!{N}(α, A::StridedData{N}, β, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int})
    quote
        if prod(dims) + prod(_filterdims(dims,C)) <= BASELENGTH
            trace_micro!(α, A, β, C, dims, offsetA, offsetC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            @dividebody $N dmax dims offsetA A offsetC C begin
                trace_rec!(α, A, β, C, dims, offsetA, offsetC, minstrides)
            end begin
                if C.strides[dmax] == 0
                    trace_rec!(α, A, _one, C, dims, offsetA, offsetC, minstrides)
                else
                    trace_rec!(α, A, β, C, dims, offsetA, offsetC, minstrides)
                end
            end
        end
        return C
    end
end

@generated function contract_rec!{N}(α, A::StridedData{N}, B::StridedData{N}, β, C::StridedData{N},
    dims::NTuple{N, Int}, offsetA::Int, offsetB::Int, offsetC::Int, minstrides::NTuple{N, Int})

    quote
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
            @dividebody $N dmax dims offsetA A offsetB B offsetC C begin
                    contract_rec!(α, A, B, β, C, dims, offsetA, offsetB, offsetC, minstrides)
                end begin
                if C.strides[dmax] == 0 # dmax is contraction dimension: β -> 1
                    contract_rec!(α, A, B, _one, C, dims, offsetA, offsetB, offsetC, minstrides)
                else
                    contract_rec!(α, A, B, β, C, dims, offsetA, offsetB, offsetC, minstrides)
                end
            end
        end
        return C
    end
end
