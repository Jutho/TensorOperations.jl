# implementation/strides.jl
#
# Implements the stride calculations of the various problems

@generated function add_strides{N}(dims::NTuple{N,Int}, stridesA::NTuple{N,Int}, stridesC::NTuple{N,Int})
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:N]...)
    quote
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute(dims, p)
        stridesA = _permute(stridesA, p)
        stridesC = _permute(stridesC, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesC, minstrides
    end
end

@generated function trace_strides{NA,NC}(dims::NTuple{NA,Int}, stridesA::NTuple{NA,Int}, stridesC::NTuple{NC,Int})
    M = div(NA-NC,2)
    dimsex = Expr(:tuple,[:(dims[$d]) for d=1:(NC+M)]...)
    stridesAex = Expr(:tuple,[:(stridesA[$d]) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    stridesCex = Expr(:tuple,[:(stridesC[$d]) for d = 1:NC]...,[0 for d = 1:M]...)
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    quote
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        newdims = _permute($dimsex, p)
        newstridesA = _permute($stridesAex, p)
        newstridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return newdims, newstridesA, newstridesC, minstrides
    end
end

@generated function contract_strides{NA, NB, NC}(dimsA::NTuple{NA, Int}, dimsB::NTuple{NB, Int},
    stridesA::NTuple{NA, Int}, stridesB::NTuple{NB, Int}, stridesC::NTuple{NC, Int})
    meta = Expr(:meta, :inline)
    cN = div(NA+NB-NC, 2)
    oNA = NA - cN
    oNB = NB - cN

    dimsex = Expr(:tuple, [:(dimsA[$d]) for d = 1:oNA]..., [:(dimsB[$d]) for d = 1:oNB]..., [:(dimsA[$(oNA+d)]) for d = 1:cN]...)

    stridesAex = Expr(:tuple, [:(stridesA[$d]) for d = 1:oNA]..., [0 for d = 1:oNB]..., [:(stridesA[$(oNA+d)]) for d = 1:cN]...)
    stridesBex = Expr(:tuple, [0 for d = 1:oNA]..., [:(stridesB[$d]) for d = 1:oNB]..., [:(stridesB[$(oNB+d)]) for d = 1:cN]...)
    stridesCex = Expr(:tuple, [:(stridesC[$d]) for d = 1:(oNA+oNB)]..., [0 for d = 1:cN]...)

    minstridesex = Expr(:tuple, [:(min(stridesA[$d], stridesC[$d])) for d = 1:oNA]...,
    [:(min(stridesB[$d], stridesC[$(oNA+d)])) for d = 1:oNB]...,
    [:(min(stridesA[$(oNA+d)], stridesB[$(oNB+d)])) for d = 1:cN]...)
    quote
        $meta
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute($dimsex, p)
        stridesA = _permute($stridesAex, p)
        stridesB = _permute($stridesBex, p)
        stridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesB, stridesC, minstrides
    end
end
