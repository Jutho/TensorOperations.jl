module TensorOperationsChainRulesCoreExt

if !isdefined(Base, :get_extension)
    using ..TensorOperations
    using ..TensorOperations: numind, numout
    using ..ChainRulesCore
else
    using TensorOperations
    using TensorOperations: numind, numout
    using ChainRulesCore
end

using TupleTools: invperm
using LinearAlgebra

_conj(conjA::Symbol) = conjA == :C ? :N : :C
trivtuple(N) = ntuple(identity, N)

@non_differentiable TensorOperations.tensorstructure(args...)
@non_differentiable TensorOperations.tensoradd_structure(args...)
@non_differentiable TensorOperations.tensoradd_type(args...)
@non_differentiable TensorOperations.tensoralloc_add(args...)
@non_differentiable TensorOperations.tensorcontract_structure(args...)
@non_differentiable TensorOperations.tensorcontract_type(args...)
@non_differentiable TensorOperations.tensoralloc_contract(args...)
@non_differentiable TensorOperations.tensorfree!(C)

# TODO: possibly use the non-inplace functions, to avoid depending on Base.copy

function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
                              C, pC::Index2Tuple,
                              A, conjA::Symbol,
                              α::Number, β::Number, backend...)
    C′ = tensoradd!(copy(C), pC, A, conjA, α, β, backend...)

    function pullback(ΔC)
        dC = @thunk conj(β) * ΔC
        dA = @thunk begin
            ipC = invperm(linearize(pC))
            c_dA = tensorcopy((ipC, ()), ΔC, conjA, conjA == :N ? conj(α) : α, backend...)
            # todo this is probably fixable with project_to
            return (!(scalartype(A) <: Complex) && (scalartype(c_dA) <: Complex)) ?
                   real(c_dA) : c_dA
        end
        dα = @thunk tensorscalar(tensorcontract(((), ()), A, ((), linearize(pC)),
                                                _conj(conjA), ΔC,
                                                (trivtuple(numind(pC)),
                                                 ()), :N, backend...))
        dβ = @thunk tensorscalar(tensorcontract(((), ()), C,
                                                ((), trivtuple(numind(pC))), :C, ΔC,
                                                (trivtuple(numind(pC)), ()), :N,
                                                backend...))
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(), dA, NoTangent(), dα, dβ, dbackend...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
                              C, pC::Index2Tuple,
                              A, pA::Index2Tuple, conjA::Symbol,
                              B, pB::Index2Tuple, conjB::Symbol,
                              α::Number, β::Number, backend...)
    C′ = tensorcontract!(copy(C), pC, A, pA, conjA, B, pB, conjB, α, β, backend...)

    function pullback(ΔC)
        dC = @thunk conj(β) * ΔC
        dA = @thunk begin
            ipC = invperm(linearize(pC))
            ipA = (invperm(linearize(pA)), ())
            NA₁ = numout(pA)
            conjΔC = conjA == :C ? :C : :N
            conjB′ = conjA == :C ? conjB : _conj(conjB)
            c_dA = tensorcontract(ipA, ΔC, (ipC[1:NA₁], ipC[(NA₁ + 1):end]), conjΔC,
                                  B, reverse(pB), conjB′, conjA == :C ? α : conj(α),
                                  backend...)
            (!(eltype(A) <: Complex) && (eltype(c_dA) <: Complex)) ? real(c_dA) : c_dA
        end
        dB = @thunk begin
            ipC = invperm(linearize(pC))
            ipB = (invperm(linearize(pB)), ())
            NA₁ = numout(pA)
            conjΔC = conjB == :C ? :C : :N
            conjA′ = conjB == :C ? conjA : _conj(conjA)

            return tensorcontract(ipB, A, reverse(pA), conjA′,
                                  ΔC, (ipC[1:NA₁], ipC[(NA₁ + 1):end]), conjΔC,
                                  conjB == :C ? α : conj(α), backend...)
            (!(eltype(B) <: Complex) && (eltype(c_dB) <: Complex)) ? real(c_dB) : c_dB
        end
        dα = @thunk tensorscalar(tensorcontract(((), ()),
                                                tensorcontract(pC, A, pA, conjA, B, pB,
                                                               conjB),
                                                ((), trivtuple(numind(pC))),
                                                :C, ΔC,
                                                (trivtuple(numind(pC)), ()), :N,
                                                backend...))
        dβ = @thunk tensorscalar(tensorcontract(((), ()), C,
                                                ((), trivtuple(numind(pC))), :C, ΔC,
                                                (trivtuple(numind(pC)), ()), :N,
                                                backend...))

        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(),
               dA, NoTangent(), NoTangent(), dB, NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end

    return C′, pullback
end

# note that this requires `one` to be defined, which is already not the case for regular
# arrays when tracing multiple indices at the same time.
function ChainRulesCore.rrule(::typeof(tensortrace!), C, pC::Index2Tuple, A,
                              pA::Index2Tuple, conjA::Symbol, α::Number, β::Number,
                              backend...)
    C′ = tensortrace!(copy(C), pC, A, pA, conjA, α, β, backend...)

    function pullback(ΔC)
        dC = @thunk conj(β) * ΔC
        dA = @thunk begin
            ipC = invperm((linearize(pC)..., pA[1]..., pA[2]...))
            E = one(TensorOperations.tensoralloc_add(scalartype(A), pA, A, conjA))
            return tensorproduct((ipC, ()), ΔC, (trivtuple(numind(pC)), ()), conjA, E,
                                 ((), trivtuple(numind(pA))), conjA,
                                 conjA == :N ? conj(α) : α, backend...)
        end
        dα = @thunk tensorscalar(tensorcontract(((), ()),
                                                tensortrace(pC, A, pA, conjA),
                                                ((), trivtuple(numind(pC))),
                                                _conj(conjA), ΔC,
                                                (trivtuple(numind(pC)), ()), :N,
                                                backend...))
        dβ = @thunk tensorscalar(tensorcontract(((), ()), C,
                                                ((), trivtuple(numind(pC))), :C, ΔC,
                                                (trivtuple(numind(pC)), ()), :N,
                                                backend...))
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(), dA, NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end

    return C′, pullback
end

end
