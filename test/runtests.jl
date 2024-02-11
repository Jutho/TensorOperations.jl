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

@testset "ad" verbose = true begin
    include("ad.jl")
end

# note: cuTENSOR should not be loaded before this point
# as there is a test which requires it to be loaded after
@testset "cuTENSOR" verbose = true begin
    include("cutensor.jl")
end

@testset "Polynomials" begin
    include("polynomials.jl")
end

@testset "Aqua" verbose = true begin
    Aqua.test_all(TensorOperations)
end
