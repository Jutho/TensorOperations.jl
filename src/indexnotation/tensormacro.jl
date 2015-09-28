# indexnotation/tensormacro.jl
#
# Defines the @tensor macro which switches to an index-notation environment.

macro tensor(arg)
    tensorify(arg)
end

function tensorify(ex::Expr)
    if ex.head == :(=) || ex.head == :(:=) || ex.head == :(+=) || ex.head == :(-=)
        lhs = ex.args[1]
        rhs = ex.args[2]
        if isa(lhs, Expr) && lhs.head == :ref
            dst = tensorify(lhs.args[1])
            src = ex.head == :(-=) ? tensorify(Expr(:call,:-,rhs)) : tensorify(rhs)
            indices = makeindex_expr(lhs)
            if ex.head == :(:=)
                return :($dst = deindexify($src, $indices))
            else
                value = ex.head == :(==) ? 0 : +1
                return :(deindexify!($dst, $src, $indices, $value))
            end
        end
    end
    if ex.head == :ref
        indices = makeindex_expr(ex)
        t = tensorify(ex.args[1])
        return :(indexify($t,$indices))
    end
    if ex.head == :call && ex.args[1] == :scalar
        if length(ex.args) != 2
            error("scalar accepts only a single argument")
        end
        src = tensorify(ex.args[2])
        indices = :(Indices{()}())
        return :(scalar(deindexify($src, $indices)))
    end
    return Expr(ex.head,map(tensorify,ex.args)...)
end
tensorify(ex::Symbol) = esc(ex)
tensorify(ex) = ex

function makeindex_expr(ex::Expr)
    if ex.head == :ref
        for i = 2:length(ex.args)
            isa(ex.args[i],Int) || isa(ex.args[i],Symbol) || isa(ex.args[i],Char) || error("cannot make indices from $ex")
        end
    else
        error("cannot make indices from $ex")
    end
    return :(Indices{$(tuple(ex.args[2:end]...))}())
end
