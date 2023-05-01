using Test
using TensorOperations
using Random

Random.seed!(1234567)

@testset "tensoropt" include("tensoropt.jl")
@testset "auxiliary" include("auxiliary.jl")

include("strided.jl")

if CUDA.functional()
    @testset "CUDA" verbose = true include("cuda.jl")
end

@testset "Polynomials" include("polynomials.jl")
