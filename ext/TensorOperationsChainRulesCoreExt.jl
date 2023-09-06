module TensorOperationsChainRulesCoreExt

if !isdefined(Base, :get_extension)
    using ..TensorOperations
    using ..TensorOperations: numind, numin, numout, promote_contract
    using ..ChainRulesCore
    using ..TupleTools
    using ..VectorInterface
else
    using TensorOperations
    using TensorOperations: numind, numin, numout, promote_contract
    using ChainRulesCore
    using TupleTools
    using VectorInterface
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

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipC = invperm(linearize(pC))
            _dA = zerovector(A, VectorInterface.promote_add(ΔC, α))
            _dA = tensoradd!(_dA, (ipC, ()), ΔC, conjA, conjA == :N ? conj(α) : α, Zero(),
                             backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            _dα = tensorscalar(tensorcontract(((), ()), A, ((), linearize(pC)),
                                              _conj(conjA), ΔC,
                                              (trivtuple(numind(pC)),
                                               ()), :N, backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            _dβ = tensorscalar(tensorcontract(((), ()), C,
                                              ((), trivtuple(numind(pC))), :C, ΔC,
                                              (trivtuple(numind(pC)), ()), :N,
                                              backend...))
            return projectβ(_dβ)
        end
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

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC)
        ipC = invperm(linearize(pC))
        pΔC = (TupleTools.getindices(ipC, trivtuple(numout(pA))),
               TupleTools.getindices(ipC, numout(pA) .+ trivtuple(numin(pB))))
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = (invperm(linearize(pA)), ())
            conjΔC = conjA == :C ? :C : :N
            conjB′ = conjA == :C ? conjB : _conj(conjB)
            _dA = zerovector(A, promote_contract(scalartype(ΔC), scalartype(B), typeof(α)))
            _dA = tensorcontract!(_dA, ipA,
                                  ΔC, pΔC, conjΔC,
                                  B, reverse(pB), conjB′,
                                  conjA == :C ? α : conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dB = @thunk begin
            ipB = (invperm(linearize(pB)), ())
            conjΔC = conjB == :C ? :C : :N
            conjA′ = conjB == :C ? conjA : _conj(conjA)
            _dB = zerovector(B, promote_contract(scalartype(ΔC), scalartype(A), typeof(α)))
            _dB = tensorcontract!(_dB, ipB,
                                  A, reverse(pA), conjA′,
                                  ΔC, pΔC, conjΔC,
                                  conjB == :C ? α : conj(α), Zero(), backend...)
            return projectB(_dB)
        end
        dα = @thunk begin
            _dα = tensorscalar(tensorcontract(((), ()),
                                              tensorcontract(pC, A, pA, conjA, B, pB,
                                                             conjB),
                                              ((), trivtuple(numind(pC))),
                                              :C, ΔC,
                                              (trivtuple(numind(pC)), ()), :N,
                                              backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            _dβ = tensorscalar(tensorcontract(((), ()), C,
                                              ((), trivtuple(numind(pC))), :C, ΔC,
                                              (trivtuple(numind(pC)), ()), :N,
                                              backend...))
            return projectβ(_dβ)
        end
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

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipC = invperm((linearize(pC)..., pA[1]..., pA[2]...))
            Es = map(pA[1], pA[2]) do i1, i2
                return one(TensorOperations.tensoralloc_add(scalartype(A), ((i1,), (i2,)),
                                                            A, conjA))
            end
            E = _kron(Es, backend...)
            _dA = zerovector(A, promote_type(scalartype(ΔC), typeof(α)))
            _dA = tensorproduct!(_dA, (ipC, ()), ΔC, (trivtuple(numind(pC)), ()), conjA, E,
                                 ((), trivtuple(numind(pA))), conjA,
                                 conjA == :N ? conj(α) : α, Zero(), backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            _dα = tensorscalar(tensorcontract(((), ()),
                                              tensortrace(pC, A, pA),
                                              ((), trivtuple(numind(pC))),
                                              _conj(conjA), ΔC,
                                              (trivtuple(numind(pC)), ()), :N,
                                              backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            _dβ = tensorscalar(tensorcontract(((), ()), C,
                                              ((), trivtuple(numind(pC))), :C, ΔC,
                                              (trivtuple(numind(pC)), ()), :N,
                                              backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(), dA, NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end

    return C′, pullback
end

_kron(Es::NTuple{1}, backend...) = Es[1]
function _kron(Es::NTuple{N,Any}, backend...) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), backend...)
    p2 = ((), trivtuple(2 * N - 2))
    p = ((1, (2 .+ trivtuple(N - 1))...), (2, ((N + 1) .+ trivtuple(N - 1))...))
    return tensorproduct(p, E1, ((1, 2), ()), :N, E2, p2, :N, One(), backend...)
end

end
