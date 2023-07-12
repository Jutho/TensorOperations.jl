using Test
using TensorOperations
using Random
using Aqua

Random.seed!(1234567)

@testset "tensoropt" begin
    include("tensoropt.jl")
end
@testset "auxiliary" begin
    include("auxiliary.jl")
end

@testset "strided" begin
    using TensorOperations
    using Test
    include("methods.jl")
    include("tensor.jl")
end

@testset "ad" begin
    include("ad.jl")
end

using CUDA
CUDA.functional() && @testset "CUDA" verbose = true begin
    include("cuda.jl")
end

@testset "Polynomials" begin
    include("polynomials.jl")
end

@testset "Aqua" verbose = true begin
    # only test project formatting for Julia >= 1.9
    Aqua.test_all(TensorOperations; stale_deps=(; ignore=[:Requires]),
                  project_toml_formatting=(VERSION >= v"1.9"))
end
