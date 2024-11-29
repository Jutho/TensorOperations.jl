using PrecompileTools: @setup_workload, @compile_workload
using Combinatorics: permutations, partitions
using Base.Iterators: flatmap

const PRECOMPILE_MAXDIMS = 4
const PRECOMPILE_ELTYPES = (Float64, ComplexF64)

function precompile_tensoradd!(A, backend=DefaultBackend(),
                               allocator=DefaultAllocator())
    @setup_workload begin
        pAs = permutations(collect(1:ndims(A)))
        conjs = (true, false)
        αs = (One(), Zero(), rand(eltype(A)))
        βs = (One(), Zero(), rand(eltype(A)))

        @compile_workload begin
            for pA in pAs, conjA in conjs
                pA′ = (tuple(pA...), ())
                C_tmp = tensoralloc_add(eltype(A), A, pA′, conjA, Val(true), allocator)
                C = tensoralloc_add(eltype(A), A, pA′, conjA, Val(false), allocator)
                for α in αs, β in βs
                    tensoradd!(C_tmp, A, pA′, conjA, α, β, backend, allocator)
                    tensoradd!(C, A, pA′, conjA, α, β, backend, allocator)
                end
                tensorfree!(C_tmp, allocator)
            end
        end
    end
    return nothing
end

function precompile_tensortrace!(A, backend=DefaultBackend(), allocator=DefaultAllocator())
    @setup_workload begin
        pqs = Iterators.filter(pq -> iseven(length(pq[2])),
                               flatmap(p -> partitions(p, 2),
                                       permutations(collect(1:ndims(A)))))

        conjs = (true, false)
        αs = (One(), Zero(), rand(eltype(A)))
        βs = (One(), Zero(), rand(eltype(A)))

        @compile_workload begin
            for pq in pqs, conjA in conjs
                p = (tuple(pq[1]...), ())
                q = (tuple(pq[2][1:2:end]...), tuple(pq[2][2:2:end]...))
                C_tmp = tensoralloc_add(eltype(A), A, p, conjA, Val(true), allocator)
                C = tensoralloc_add(eltype(A), A, p, conjA, Val(false), allocator)
                for α in αs, β in βs
                    tensortrace!(C_tmp, A, p, q, conjA, α, β, backend, allocator)
                    tensortrace!(C, A, p, q, conjA, α, β, backend, allocator)
                end
                tensorfree!(C_tmp, allocator)
            end
        end
    end
end

function precompile_tensorcontract!(A, B, backend=DefaultBackend(),
                                    allocator=DefaultAllocator())
    @setup_workload begin
        conjAs = (true, false)
        conjBs = (true, false)
        αs = (One(), Zero(), rand(eltype(A)))
        βs = (One(), Zero(), rand(eltype(A)))
        pAs = permutations(collect(1:ndims(A)))
        pBs = permutations(collect(1:ndims(B)))
        @compile_workload begin
            TC = promote_contract(eltype(A), eltype(B))
            for pA′ in pAs, pB′ in pBs
                for k in 0:min(ndims(A), ndims(B))
                    pA = (tuple(pA′[1:(end - k)]...), tuple(pA′[(end - k + 1):end]...))
                    pB = (tuple(pB′[1:k]...), tuple(pB′[(k + 1):end]...))
                    for pAB′ in permutations(collect(1:(ndims(A) + ndims(B) - 2k)))
                        pAB = (tuple(pAB′...), ())
                        for conjA in conjAs, conjB in conjBs
                            C_tmp = tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB,
                                                         pAB,
                                                         Val(true), allocator)
                            C = tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB,
                                                     Val(false), allocator)
                            for α in αs, β in βs
                                tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
                                                backend, allocator)
                                tensorcontract!(C_tmp, A, pA, conjA, B, pB, conjB, pAB, α,
                                                β,
                                                backend, allocator)
                            end
                            tensorfree!(C_tmp, allocator)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

for backend in (StridedBLAS(),),
    allocator in (DefaultAllocator(), ManualAllocator())

    for NA in 1:PRECOMPILE_MAXDIMS, TA in PRECOMPILE_ELTYPES
        A = Array{TA}(undef, ntuple(Returns(1), NA))
        precompile_tensoradd!(A, backend, allocator)
        precompile_tensortrace!(A, backend, allocator)
    end
    for NA in 1:PRECOMPILE_MAXDIMS, NB in 1:PRECOMPILE_MAXDIMS, TA in PRECOMPILE_ELTYPES
        A = Array{TA}(undef, ntuple(Returns(1), NA))
        B = Array{TA}(undef, ntuple(Returns(1), NB))
        precompile_tensorcontract!(A, B, backend, allocator)
    end
end
