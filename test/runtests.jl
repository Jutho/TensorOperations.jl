using Test
using TensorOperations
using Random

Random.seed!(1234567)

@testset "tensoropt" begin
    include("tensoropt.jl")
end
@testset "auxiliary" begin
    include("auxiliary.jl")
end

include("strided.jl")

include("ad.jl")

using CUDA
CUDA.functional() && @testset "CUDA" verbose = true begin
    include("cuda.jl")
end

@testset "Polynomials" begin
    include("polynomials.jl")
end
