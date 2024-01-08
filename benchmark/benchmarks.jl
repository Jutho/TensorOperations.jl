using BenchmarkTools
using TensorOperations
using TensorOperations: StridedBLAS, Backend
using MKL, ThreadPinning, LinearAlgebra
using TensorOperationsTBLIS
using cuTENSOR

# Setup
# -----

SUITE = BenchmarkGroup()

nt = Threads.nthreads()
BLAS.set_num_threads(nt)
TensorOperationsTBLIS.tblis_set_num_threads(nt)

ThreadPinning.mkl_set_dynamic(false)
pinthreads(:cores)
threadinfo(; blas=true, hints=true)

# Additions
# ---------

SUITE["tensoradd!"] = BenchmarkGroup(["methods"])

# parameters
PERMUTATIONS_SPECS = joinpath(@__DIR__, "permutations.dat")
isfile(PERMUTATIONS_SPECS) ||
    error("additions specs file ($PERMUTATIONS_SPECS) not found")

Ts = (Float64, ComplexF64)
backends = (StridedBLAS(), Backend{:cuTENSOR}(), Backend{:tblis}())

# utility functions
function extract_permute_labels(permutation::AbstractString)
    symbolsC = match(r"C\[([^\]]*)\]", permutation)
    labelsC = split(symbolsC.captures[1], ","; keepempty=false)
    symbolsA = match(r"A\[([^\]]*)\]", permutation)
    labelsA = split(symbolsA.captures[1], ","; keepempty=false)
    return labelsC, labelsA
end

function generate_permute_benchmark(line::AbstractString; T=Float64,
                                    backend=TensorOperations.StridedBLAS())
    permutation, sizes = split(line, " & ")

    # extract labels
    labelsC, labelsA = extract_permute_labels(permutation)
    pC = TensorOperations.add_indices(tuple(labelsA...), tuple(labelsC...))

    # extract sizes
    subsizes = Dict{String,Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
        subsizes[label] = parse(Int, sz)
    end
    szA = getindex.(Ref(subsizes), labelsA)
    szC = getindex.(Ref(subsizes), labelsC)
    α = rand(T)
    β = rand(T)

    return @benchmarkable(tensoradd!(C, $pC, A, :N, $α, $β, $backend),
                          setup = (A = rand($T, $szA...);
                                   C = rand($T, $szC...)),
                          evals = 1)
end

for T in Ts
    tags = T <: Complex ? ["complex"] : ["real"]
    SUITE["tensoradd!"][T] = BenchmarkGroup(tags)

    for backend in backends
        SUITE["tensoradd!"][T][backend] = BenchmarkGroup([string(backend)])

        for (i, line) in enumerate(eachline(PERMUTATIONS_SPECS))
            SUITE["tensoradd!"][T][backend][i] = generate_permute_benchmark(line; T,
                                                                            backend)
            i >= 3 && break
        end
    end
end

# Contractions
# ------------

SUITE["tensorcontract!"] = BenchmarkGroup(["methods"])

# parameters
CONTRACTIONS_SPECS = joinpath(@__DIR__, "contractions.dat")
isfile(CONTRACTIONS_SPECS) ||
    error("contractions specs file ($CONTRACTIONS_SPECS) not found")

Ts = (Float64, ComplexF64)
backends = (StridedBLAS(), Backend{:cuTENSOR}(), Backend{:tblis}())

# utility functions
function extract_contract_labels(contraction::AbstractString)
    symbolsC = match(r"C\[([^\]]*)\]", contraction)
    labelsC = split(symbolsC.captures[1], ","; keepempty=false)
    symbolsA = match(r"A\[([^\]]*)\]", contraction)
    labelsA = split(symbolsA.captures[1], ","; keepempty=false)
    symbolsB = match(r"B\[([^\]]*)\]", contraction)
    labelsB = split(symbolsB.captures[1], ","; keepempty=false)
    return labelsC, labelsA, labelsB
end

function generate_contract_benchmark(line::AbstractString;
                                     T=Float64, backend=TensorOperations.StridedBLAS())
    contraction, sizes = split(line, " & ")

    # extract labels
    labelsC, labelsA, labelsB = extract_contract_labels(contraction)
    pA, pB, pC = TensorOperations.contract_indices(tuple(labelsA...), tuple(labelsB...),
                                                   tuple(labelsC...))

    # extract sizes
    subsizes = Dict{String,Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
        subsizes[label] = parse(Int, sz)
    end
    szA = getindex.(Ref(subsizes), labelsA)
    szB = getindex.(Ref(subsizes), labelsB)
    szC = getindex.(Ref(subsizes), labelsC)
    α = rand(T)
    β = rand(T)

    return @benchmarkable(tensorcontract!(C, $pC, A, $pA, :N, B, $pB, :N, $α, $β, $backend),
                          setup = (A = rand($T, $szA...);
                                   B = rand($T, $szB...);
                                   C = rand($T, $szC...)),
                          evals = 1)
end

for T in Ts
    tags = T <: Complex ? ["complex"] : ["real"]
    SUITE["tensorcontract!"][T] = BenchmarkGroup(tags)

    for backend in backends
        SUITE["tensorcontract!"][T][backend] = BenchmarkGroup([string(backend)])

        for (i, line) in enumerate(eachline(CONTRACTIONS_SPECS))
            SUITE["tensorcontract!"][T][backend][i] = generate_contract_benchmark(line; T,
                                                                                  backend)
            i >= 3 && break
        end
    end
end
