if get(ENV, "MKL", "true") == "true"
    using MKL
end

strided_threads = parse(Int, get(ENV, "STRIDED_THREADS", 1))
blas_threads = parse(Int, get(ENV, "BLAS_THREADS", 1))

using Strided: Strided
using LinearAlgebra: BLAS
Strided.set_num_threads(strided_threads)
BLAS.set_num_threads(blas_threads)

using BenchmarkTools
using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, Backend
using cuTENSOR
using LinearAlgebra

## Setup Parameters
# ------------------

const SUITE = BenchmarkGroup()

include("benchmark_utils.jl")

const Ts = (Float32, Float64, ComplexF32, ComplexF64)
const backends = (StridedBLAS(), StridedNative(), Backend{:cuTENSOR}())
const max_tests = Inf

## Utility functions
# -------------------

## Additions
# -----------
add_suite = SUITE["tensoradd!"] = BenchmarkGroup(["methods"])

for T in Ts
    tags = T <: Complex ? ["complex"] : ["real"]
    add_suite[T] = BenchmarkGroup(tags)

    for backend in backends
        backend == StridedBLAS() && continue # same as StridedNative
        backendsuite = add_suite[T][backend] = BenchmarkGroup([string(backend)])

        for (i, line) in enumerate(eachline(PERMUTATIONS_SPECS))
            backendsuite[i] = generate_permute_benchmark(line; T, backend)
            i >= max_tests && break
        end
    end
end

## Contractions
# --------------
contract_suite = SUITE["tensorcontract!"] = BenchmarkGroup(["methods"])

for T in Ts
    tags = T <: Complex ? ["complex"] : ["real"]
    contract_suite[T] = BenchmarkGroup(tags)

    for backend in backends
        backendsuite = contract_suite[T][backend] = BenchmarkGroup([string(backend)])

        for (i, line) in enumerate(eachline(CONTRACTIONS_SPECS))
            backendsuite[i] = generate_contract_benchmark(line; T, backend)
            i >= max_tests && break
        end
    end
end
