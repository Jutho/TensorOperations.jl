# gradients/flux.jl
#
# Connect up gradients for Zygote?

using .Zygote
using .Zygote: @adjoint, @nograd

@nograd similar_from_indices, cached_similar_from_indices
@nograd dirac, dirac!


@adjoint function add!(α, A, conjA, β, C, indCinA)
	∇VERBOSE && @info "@adjoint add!"
    add!(α, A, conjA, β, C, indCinA),
        Δ -> ∇add(Δ, α, A, conjA, β, C, indCinA)
end

@adjoint function trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)
	∇VERBOSE && @info "@adjoint trace!"
    trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2),
        Δ -> ∇trace(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) 
end

@adjoint function contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
	∇VERBOSE && @info "@adjoint contract!"
    contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms),
        Δ -> ∇contract(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms) 
end

# It's currently not possible to skip the calculation of un-needed gradients, as done in Flux case

function ∇add(Δ, α::Tα, A::TA, conjA, β::Tβ, C::TC, indCinA) where {Tα,TA,Tβ,TC}

    ∇A = add∇A(Δ, α, A, conjA, β, C, indCinA)
    ∇C = any∇C(Δ,β)

    ∇α = 0 # false 
    ∇β = 0 # false

    return (∇α, ∇A, nothing, ∇β, ∇C, nothing)
end

function ∇trace(Δ, α::Tα, A::TA, conjA, β::Tβ, C::TC, indCinA, cindA1, cindA2) where {Tα,TA,Tβ,TC}

    ∇A = trace∇A(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) 
    ∇C = any∇C(Δ,β)

    ∇α = 0 # false # trace∇α(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) 
    ∇β = 0 # false # trace∇β(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) 

    return (∇α, ∇A, nothing, ∇β, ∇C, nothing, nothing, nothing)
end

function ∇contract(Δ, α::Tα, A::TA, conjA, B::TB, conjB, β::Tβ, C::TC, oindA, cindA, oindB, cindB, indCinoAB, syms) where {Tα,TA,Tβ,TB,TC}

    ∇A = contract∇A(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    ∇B = contract∇B(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    ∇C = any∇C(Δ,β)

    ∇α = 0 # false # contract∇α(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB)
    ∇β = 0 # false # contract∇β(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB)

    return (∇α, ∇A, nothing, ∇B, nothing, ∇β, ∇C, nothing, nothing, nothing, nothing, nothing, nothing)
end
