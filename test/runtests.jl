using Test
using TensorOperations
using Random
using Aqua

Random.seed!(1234567)

@testset "tensoropt" begin
    include("tensoropt.jl")
end
@testset "auxiliary" verbose = true begin
    include("auxiliary.jl")
end

@testset "macro keywords" verbose = true begin
    include("macro_kwargs.jl")
end

@testset "strided" verbose = true begin
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
    Aqua.test_all(TensorOperations;
                  project_toml_formatting=(VERSION >= v"1.9"))
end
