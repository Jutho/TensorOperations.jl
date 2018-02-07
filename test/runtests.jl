@static if VERSION < v"0.7-"
    const Test = Base.Test
    const ComplexF32 = Complex64
    const ComplexF64 = Complex128
else
    using Random
    using LinearAlgebra
end

# Until problems with .+ transforming ranges to arrays are settled
⊞(s::Int, r::StepRange{Int,Int}) = (first(r)+s):step(r):(last(r)+s)
⊞(s::Int, r::UnitRange{Int}) = (first(r)+s):(last(r)+s)

using Test
using TensorOperations

include("methods.jl")
include("tensor.jl")
include("tensoropt.jl")
