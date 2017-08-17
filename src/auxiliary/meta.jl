# auxiliary/meta.jl
#
# A bunch of auxiliary metaprogramming tools and generated functions

@generated function _strides(A::StridedArray{T,N}) where {T,N}
    meta = Expr(:meta,:inline)
    ex = Expr(:tuple,[:(stride(A,$d)) for d = 1:N]...)
    Expr(:block, meta, ex)
end

@generated function _indmax(values::NTuple{N,T}) where {N,T}
    meta = Expr(:meta,:inline)
    Expr(:block, meta, :(dmax = 1), :(max = values[1]), [:(values[$d] > max && (dmax = $d; max = values[$d])) for d = 2:N]..., :(return dmax))
end

@generated function _permute(t::NTuple{N,T}, p) where {T,N}
    meta = Expr(:meta,:inline)
    ex = Expr(:tuple,[:(t[p[$d]]) for d = 1:N]...)
    Expr(:block, meta, ex)
end

@generated function _memjumps(dims::NTuple{N,Int},strides::NTuple{N,Int}) where N
    meta = Expr(:meta,:inline)
    ex = Expr(:tuple,[:((dims[$d]-1)*strides[$d]) for d = 1:N]...)
    Expr(:block, meta, ex)
end

# Based on Tim Holy's Cartesian
function _sreplace(ex::Expr, s::Symbol, v)
    Expr(ex.head,[_sreplace(a, s, v) for a in ex.args]...)
end
_sreplace(ex::Symbol, s::Symbol, v) = ex == s ? v : ex
_sreplace(ex, s::Symbol, v) = ex

macro dividebody(N, dmax, dims, args...)
    esc(_dividebody(N, dmax, dims, args...))
end

function _dividebody(N::Int, dmax::Symbol, dims::Symbol, args...)
    mod(length(args),2)==0 || error("Wrong number of arguments")
    argiter = 1:2:length(args)-2

    ex = Expr(:block)
    newdims = gensym(:newdims)
    newdim = gensym(:newdim)
    mainex1 = _sreplace(args[end-1], dims, newdims)
    mainex2 = _sreplace(args[end], dims, newdims)

    for d = 1:N
        updateex = Expr(:block,[:($(args[i]) += $newdim*$(args[i+1]).strides[$d]) for i in argiter]...)
        newdimsex = Expr(:tuple,[Expr(:ref,dims,i) for i=1:d-1]..., newdim, [Expr(:ref,dims,i) for i=d+1:N]...)
        body = quote
            $newdim = $dims[$d] >> 1
            $newdims = $newdimsex
            $mainex1
            $updateex
            $newdim = $dims[$d] - $newdim
            $newdims = $newdimsex
            $mainex2
        end
        ex = Expr(:if,:($dmax == $d), body,ex)
    end
    ex
end

macro stridedloops(N, dims, args...)
    esc(_stridedloops(N, dims, args...))
end
function _stridedloops(N::Int, dims::Symbol, args...)
    mod(length(args),3)==1 || error("Wrong number of arguments")
    argiter = 1:3:length(args)-1
    body = args[end]
    pre = [Expr(:(=), args[i], Symbol(args[i],0)) for i in argiter]
    ex = Expr(:block, pre..., body)
    for d = 1:N
        pre = [Expr(:(=), Symbol(args[i], d-1), Symbol(args[i], d)) for i in argiter]
        post = [Expr(:(+=), Symbol(args[i], d), Expr(:ref, args[i+2], d)) for i in argiter]
        ex = Expr(:block, pre..., ex, post...)
        rangeex = Expr(:(:), 1, Expr(:ref, dims, d))
        forex = Expr(:(=), gensym(), rangeex)
        ex = Expr(:for, forex, ex)
        if d==1
            if VERSION < v"0.7-"
                ex = Expr(:macrocall, Symbol("@simd"), ex)
            else
                ex = Expr(:macrocall, Symbol("@simd"), LineNumberNode(@__LINE__), ex)
            end
        end
    end
    pre = [Expr(:(=),Symbol(args[i],N),args[i+1]) for i in argiter]
    ex = Expr(:block, pre..., ex)
    return ex
end
