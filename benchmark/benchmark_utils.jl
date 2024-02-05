using BenchmarkTools
import Printf: @sprintf
using Dates: Dates

const PERMUTATIONS_SPECS = joinpath(@__DIR__, "benchmark_specs", "permutations.dat")
const CONTRACTIONS_SPECS = joinpath(@__DIR__, "benchmark_specs", "contractions.dat")

isfile(PERMUTATIONS_SPECS) ||
    error("additions specs file ($PERMUTATIONS_SPECS) not found")
isfile(CONTRACTIONS_SPECS) ||
    error("contractions specs file ($CONTRACTIONS_SPECS) not found")
    
function result_filename(id::String, use_mkl::Bool, blas_threads::Int, strided_threads::Int)
    blas_vendor = use_mkl ? "MKL" : "OpenBLAS"
    fn = @sprintf "%s_%s=%d_Strided=%d.json" id blas_vendor blas_threads strided_threads
    return joinpath(@__DIR__, "results", fn)
end

function extract_permute_labels(permutation::AbstractString)
    symbolsC = match(r"C\[([^\]]*)\]", permutation)
    labelsC = split(symbolsC.captures[1], ","; keepempty=false)
    symbolsA = match(r"A\[([^\]]*)\]", permutation)
    labelsA = split(symbolsA.captures[1], ","; keepempty=false)
    return labelsC, labelsA
end

function compute_permute_size(line::AbstractString)
    permutation, sizes = split(line, " & ")

    # extract labels
    labelsC, labelsA = extract_permute_labels(permutation)

    # extract sizes
    subsizes = Dict{String,Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
        subsizes[label] = parse(Int, sz)
    end
    szA = getindex.(Ref(subsizes), labelsA)
    return prod(szA)
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

function compute_contract_ops(line::AbstractString)
    contraction, sizes = split(line, " & ")

    # extract labels
    labelsC, labelsA, labelsB = extract_contract_labels(contraction)
    pA, pB, = TensorOperations.contract_indices(tuple(labelsA...), tuple(labelsB...),
                                                   tuple(labelsC...))
    # extract sizes
    subsizes = Dict{String,Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
        subsizes[label] = parse(Int, sz)
    end
    szA = getindex.(Ref(subsizes), labelsA)
    szB = getindex.(Ref(subsizes), labelsB)
    szC = getindex.(Ref(subsizes), labelsC)
    return prod(getindex.(Ref(szA), pA[1])) * prod(getindex.(Ref(szA), pA[2])) *
           prod(getindex.(Ref(szA), pA[2]))
end