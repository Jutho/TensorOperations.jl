abstract type TensorExpr end

struct LinearCombination <: TensorExpr
    terms
    stype
end
struct GeneralTensor <: TensorExpr
    object
    leftind
    rightind
    scalar
    conj
    stype
end
struct TensorContraction <: TensorExpr
    factor1
    factor2
    scalar
    scalartype
end
struct ScalarExpr
    expr
    scalartype
end

function Base.:*(s₁::ScalarExpr, s₂::ScalarExpr)
    scalartype = Expr(:call, :(Base.promote_op), :*, s₁.scalartype, s₂.scalartype)
    scalar = Expr(:call, :*, s₁, s₂)
    return ScalarExpr(scalar, scalartype)
end
function Base.:*(s::ScalarExpr, t::GeneralTensor)
    scalartype = Expr(:call, :(Base.promote_op), :*, s.scalartype, t.scalartype)
    scalar = t.scalar === true ? s : Expr(:call, :*, s, t.scalar)
    return GeneralTensor(t.object, t.leftind, t.rightind, scalar, t.conj, scalartype)
end
function Base.:*(t::GeneralTensor, s::ScalarExpr)
    scalartype = Expr(:call, :(Base.promote_op), :*, t.scalartype, s.scalartype)
    scalar = t.scalar === true ? s : Expr(:call, :*, s, t.scalar)
    return GeneralTensor(t.object, t.leftind, t.rightind, scalar, t.conj, scalartype)
end
function Base.:*(s::ScalarExpr, t::LinearCombination)
    factors = s .* t.factors
    scalartype = Expr(:call, :(Base.promote_op), :*, s.scalartype, t.scalartype)
    return LinearCombination(factors, scalartype)
end
function Base.:*(t::LinearCombination, s::ScalarExpr)
    factors = t.factors .* s
    scalartype = Expr(:call, :(Base.promote_op), :*, t.scalartype, s.scalartype)
    return LinearCombination(factors, scalartype)
end
function Base.:*(s::ScalarExpr, t::TensorContraction)
    scalar = (t.scalar === true) ? s : s * t.scalar
    scalartype = (t.scalar === true) ? s.scalartype : Expr(:call, :(Base.promote_op), :*, s.scalartype, t.scalartype)
end
function Base.:*(t::TensorContraction, s::ScalarExpr)
    scalar = (t.scalar === true) ? s : t.scalar * s
    scalartype = (t.scalar === true) ? s.scalartype : Expr(:call, :(Base.promote_op), :*, t.scalartype, s.scalartype)
end
function Base.:*(t₁::GeneralTensor, t₂::GeneralTensor)
    
    scalar = (t.scalar === true) ? s : s * t.scalar
    scalartype = (t.scalar === true) ? s.scalartype : Expr(:call, :(Base.promote_op), :*, s.scalartype, t.scalartype)
end
function Base.:*(t::TensorContraction, s::ScalarExpr)
    scalar = (t.scalar === true) ? s : t.scalar * s
    scalartype = (t.scalar === true) ? s.scalartype : Expr(:call, :(Base.promote_op), :*, t.scalartype, s.scalartype)
end



ScalarExpr(s::Number) = ScalarExpr(s, scalartype(s))
ScalarExpr(ex::Expr) = ScalarExpr(ex, Expr(:call, :scalartype, ex))




function GeneralTensor(ex::Expr)
    object, leftind, rightind, scalar, conj = decomposegeneraltensor(ex)
    if scalar === true
        return GeneralTensor(object, leftind, rightind, scalar, conj, :(scalartype($object)))
    else
        return GeneralTensor(object, leftind, rightind, scalar, conj, :(Base.promote_op(:*, scalartype($scalar), scalartype($object))))
    end
end
function LinearCombination(ex::Expr)
    args = ex.args
    if ex.args[1] == :- && length(ex.args) == 3
        factors = [TensorExpr(args[2]), TensorExpr(Expr(:call, :-, args[3]))]
        stype = Expr(:call, :(Base.promote_op), :+, factors[1].stype, factors[2].stype)
        return LinearCombination(factors, stype)
    elseif ex.args[1] == :+
        factors = map(TensorExpr, args[2:end])
        stype = Expr(:call, :(Base.promote_op), :+, [f.stype for f in factors]...)
        return LinearCombination(factors, stype)
    else
        throw(ArgumentError("Cannot parse the linear combination $ex"))
    end
end
function TensorContraction(ex:Expr)



function buildgraph(ex::Expr)
    if isgeneraltensor(ex)

    end
end
