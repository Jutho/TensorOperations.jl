using TensorOperations
using LinearAlgebra
using Test
using Random
using Aqua

Random.seed!(1234567)

include("tensoropt.jl")
include("auxiliary.jl")
include("macro_kwargs.jl")
include("methods.jl")
include("tensor.jl")
include("ad.jl")

# note: cuTENSOR should not be loaded before this point
# as there is a test which requires it to be loaded after
@testset "cuTENSOR" verbose = true begin
    include("cutensor.jl")
end

@testset "Polynomials" begin
    include("polynomials.jl")
end

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(TensorOperations)
end
