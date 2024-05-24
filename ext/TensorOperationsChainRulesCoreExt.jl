module TensorOperationsChainRulesCoreExt

using TensorOperations
using TensorOperations: numind, numin, numout, promote_contract, Backend
using ChainRulesCore
using TupleTools
using VectorInterface
using TupleTools: invperm
using LinearAlgebra

trivtuple(N) = ntuple(identity, N)

@non_differentiable TensorOperations.tensorstructure(args...)
@non_differentiable TensorOperations.tensoradd_structure(args...)
@non_differentiable TensorOperations.tensoradd_type(args...)
@non_differentiable TensorOperations.tensoralloc_add(args...)
@non_differentiable TensorOperations.tensorcontract_structure(args...)
@non_differentiable TensorOperations.tensorcontract_type(args...)
@non_differentiable TensorOperations.tensoralloc_contract(args...)

# Cannot free intermediate tensors when using AD
# Thus we change the forward passes: `istemp=false` and `tensorfree!` is a no-op
function ChainRulesCore.rrule(::typeof(TensorOperations.tensorfree!), args...)
    tensorfree!_pullback(Δargs...) = ntuple(x -> NoTangent(), length(args))
    return nothing, tensorfree!_pullback
end
function ChainRulesCore.rrule(::typeof(TensorOperations.tensoralloc), ttype, structure,
                              istemp, backend...)
    output = TensorOperations.tensoralloc(ttype, structure, false, backend...)
    tensoralloc_pullback(Δargs...) = ntuple(x -> NoTangent(), 4 + length(backend))
    return output, tensoralloc_pullback
end

# TODO: possibly use the non-inplace functions, to avoid depending on Base.copy

function ChainRulesCore.rrule(::typeof(tensorscalar), C)
    function tensorscalar_pullback(Δc)
        ΔC = TensorOperations.tensoralloc(typeof(C), TensorOperations.tensorstructure(C))
        return NoTangent(), fill!(ΔC, unthunk(Δc))
    end
    return tensorscalar(C), tensorscalar_pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
                              C,
                              A, pA::Index2Tuple, conjA::Bool,
                              α::Number, β::Number, backend::Backend...)
    C′ = tensoradd!(copy(C), A, pA, conjA, α, β, backend...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = invperm(linearize(pA))
            _dA = zerovector(A, VectorInterface.promote_add(ΔC, α))
            _dA = tensoradd!(_dA, ΔC, (ipA, ()), conjA, conjA ? α : conj(α), Zero(),
                             backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            _dα = tensorscalar(tensorcontract(A, ((), linearize(pA)), !conjA,
                                              ΔC, (trivtuple(numind(pA)), ()), false,
                                              ((), ()), One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            # TODO: consider using `inner`
            _dβ = tensorscalar(tensorcontract(C, ((), trivtuple(numind(pA))), true,
                                              ΔC, (trivtuple(numind(pA)), ()), false,
                                              ((), ()), One(), backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dbackend...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
                              C,
                              A, pA::Index2Tuple, conjA::Bool,
                              B, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple,
                              α::Number, β::Number, backend::Backend...)
    C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, backend...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = (TupleTools.getindices(ipAB, trivtuple(numout(pA))),
               TupleTools.getindices(ipAB, numout(pA) .+ trivtuple(numin(pB))))
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = (invperm(linearize(pA)), ())
            conjΔC = conjA
            conjB′ = conjA ? conjB : !conjB
            _dA = zerovector(A, promote_contract(scalartype(ΔC), scalartype(B), typeof(α)))
            _dA = tensorcontract!(_dA,
                                  ΔC, pΔC, conjΔC,
                                  B, reverse(pB), conjB′,
                                  ipA,
                                  conjA ? α : conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dB = @thunk begin
            ipB = (invperm(linearize(pB)), ())
            conjΔC = conjB
            conjA′ = conjB ? conjA : !conjA
            _dB = zerovector(B, promote_contract(scalartype(ΔC), scalartype(A), typeof(α)))
            _dB = tensorcontract!(_dB,
                                  A, reverse(pA), conjA′,
                                  ΔC, pΔC, conjΔC,
                                  ipB,
                                  conjB ? α : conj(α), Zero(), backend...)
            return projectB(_dB)
        end
        dα = @thunk begin
            C_αβ = tensorcontract(A, pA, conjA,
                                  B, pB, conjB,
                                  pAB, One(), backend...)
            # TODO: consider using `inner`
            _dα = tensorscalar(tensorcontract(C_αβ, ((), trivtuple(numind(pAB))), true,
                                              ΔC, (trivtuple(numind(pAB)), ()), false,
                                              ((), ()), One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            # TODO: consider using `inner`
            _dβ = tensorscalar(tensorcontract(C, ((), trivtuple(numind(pAB))), true,
                                              ΔC, (trivtuple(numind(pAB)), ()), false,
                                              ((), ()), One(), backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC,
               dA, NoTangent(), NoTangent(), dB, NoTangent(), NoTangent(),
               NoTangent(), dα, dβ, dbackend...
    end

    return C′, pullback
end

# note that this requires `one` to be defined, which is already not the case for regular
# arrays when tracing multiple indices at the same time.
function ChainRulesCore.rrule(::typeof(tensortrace!), C,
                              A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                              α::Number, β::Number, backend::Backend...)
    C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, backend...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ip = invperm((linearize(p)..., q[1]..., q[2]...))
            Es = map(q[1], q[2]) do i1, i2
                return one(TensorOperations.tensoralloc_add(scalartype(A), A,
                                                            ((i1,), (i2,)), conjA))
            end
            E = _kron(Es, backend...)
            _dA = zerovector(A, VectorInterface.promote_scale(ΔC, α))
            _dA = tensorproduct!(_dA, ΔC, (trivtuple(numind(p)), ()), conjA,
                                 E, ((), trivtuple(numind(q))), conjA,
                                 (ip, ()),
                                 conjA ? α : conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            C_αβ = tensortrace(A, p, q, false, One(), backend...)
            _dα = tensorscalar(tensorcontract(C_αβ, ((), trivtuple(numind(p))),
                                              !conjA,
                                              ΔC, (trivtuple(numind(p)), ()), false,
                                              ((), ()), One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            _dβ = tensorscalar(tensorcontract(C, ((), trivtuple(numind(p))), true,
                                              ΔC, (trivtuple(numind(p)), ()), false,
                                              ((), ()), One(), backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end

    return C′, pullback
end

_kron(Es::NTuple{1}, backend::Backend...) = Es[1]
function _kron(Es::NTuple{N,Any}, backend::Backend...) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), backend...)
    p2 = ((), trivtuple(2 * N - 2))
    p = ((1, (2 .+ trivtuple(N - 1))...), (2, ((N + 1) .+ trivtuple(N - 1))...))
    return tensorproduct(p, E1, ((1, 2), ()), false, E2, p2, false, One(), backend...)
end

# NCON functions
@non_differentiable TensorOperations.ncontree(args...)
@non_differentiable TensorOperations.nconoutput(args...)
@non_differentiable TensorOperations.isnconstyle(args...)
@non_differentiable TensorOperations.indexordertree(args...)

end
