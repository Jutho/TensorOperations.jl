using TensorOperations
using LinearAlgebra
using Test
using TestExtras
using Random
using Aqua

Random.seed!(1234567)
@timedtestset "tensoropt" verbose = true begin
    include("tensoropt.jl")
end
@timedtestset "auxiliary" verbose = true begin
    include("auxiliary.jl")
end
@timedtestset "macro keywords" verbose = true begin
    include("macro_kwargs.jl")
end
@timedtestset "method syntax" verbose = true begin
    include("methods.jl")
end
@timedtestset "macro with index notation" verbose = true begin
    include("tensor.jl")
end
@timedtestset "ad" verbose = false begin
    include("ad.jl")
end

# note: cuTENSOR should not be loaded before this point
# as there is a test which requires it to be loaded after
@timedtestset "cuTENSOR" verbose = true begin
    include("cutensor.jl")
end

@timedtestset "Polynomials" begin
    include("polynomials.jl")
end

@timedtestset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(TensorOperations)
end
