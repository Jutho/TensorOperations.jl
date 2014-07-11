# extend cartesian with extra functionality

macro ngenerate(itersym, returntypeexpr, Nlist, funcexpr)
    isfuncexpr(funcexpr) || error("Requires a function expression")
    esc(ngenerate(itersym, returntypeexpr, funcexpr.args[1], N->intarith!(sreplace!(copy(funcexpr.args[2]), itersym, N)), eval(Nlist)))
end
macro mngenerate(itersym1, itersym2, returntypeexpr, MNlist, funcexpr)
    isfuncexpr(funcexpr) || error("Requires a function expression")
    esc(mngenerate(itersym1, itersym2, returntypeexpr, funcexpr.args[1], (M,N)->intarith!(sreplace!(sreplace!(copy(funcexpr.args[2]), itersym1, M), itersym2, N)), eval(MNlist)))
end
macro mnpgenerate(itersym1, itersym2, itersym3, returntypeexpr, MNPlist, funcexpr)
    isfuncexpr(funcexpr) || error("Requires a function expression")
    esc(mnpgenerate(itersym1, itersym2, itersym3, returntypeexpr, funcexpr.args[1], (M,N,P)->intarith!(sreplace!(sreplace!(sreplace!(copy(funcexpr.args[2]), itersym1, M), itersym2, N), itersym3, P)), eval(MNPlist)))
end

generate1(itersym, prototype, bodyfunc, N::Int) =
    Expr(:function, spliceint!(sreplace!(copy(prototype), itersym, N)), bodyfunc(N))
generate2(itersym1, itersym2, prototype, bodyfunc, M::Int, N::Int) =
    Expr(:function, spliceint!(sreplace!(sreplace!(copy(prototype), itersym1, M), itersym2, N)), bodyfunc(M,N))
generate3(itersym1, itersym2, itersym3, prototype, bodyfunc, M::Int, N::Int, P::Int) =
    Expr(:function, spliceint!(sreplace!(sreplace!(sreplace!(copy(prototype), itersym1, M), itersym2, N), itersym3, P)), bodyfunc(M,N,P))

function ngenerate(itersym, returntypeexpr, prototype, bodyfunc, Nlist)
    # Generate versions for specific dimensions
    fdim = [generate1(itersym, prototype, bodyfunc, N) for N in Nlist]

    # Generate the generic cache-based version
    fsym = funcsym(prototype)
    dictname = symbol(string(fsym)*"_cache")
    fargs = funcargs(prototype)
    flocal = funcrename(copy(prototype), :_F_)
    F = Expr(:function, prototype, quote
             if !haskey($dictname, $itersym)
                 gen1 = generate1($(symbol(itersym)), $(Expr(:quote, flocal)), $bodyfunc, $itersym)
                 $(dictname)[$itersym] = eval(quote
                     local _F_
                     $gen1
                     _F_
                 end)
             end
             ($(dictname)[$itersym]($(fargs...)))::$returntypeexpr
         end)
    Expr(:block, fdim..., quote
            let $dictname = Dict{Int,Function}()
            $F
            end
        end)
end
function mngenerate(itersym1, itersym2, returntypeexpr, prototype, bodyfunc, MNlist)
    # Generate versions for specific dimensions
    fdim = [generate2(itersym1, itersym2, prototype, bodyfunc, M, N) for (M,N) in MNlist]
    
    # Generate the generic cache-based version
    fsym = funcsym(prototype)
    dictname = symbol(string(fsym)*"_cache")
    fargs = funcargs(prototype)
    flocal = funcrename(copy(prototype), :_F_)
    F = Expr(:function, prototype, quote
             if !haskey($dictname, ($itersym1, $itersym2))
                 gen2 = generate2($(symbol(itersym1)), $(symbol(itersym2)), $(Expr(:quote, flocal)), $bodyfunc, $itersym1, $itersym2)
                 $(dictname)[($itersym1, $itersym2)] = eval(quote
                     local _F_
                     $gen2
                     _F_
                 end)
             end
             ($(dictname)[($itersym1, $itersym2)]($(fargs...)))::$returntypeexpr
         end)
    Expr(:block, fdim..., quote
            let $dictname = Dict{(Int,Int),Function}()
            $F
            end
        end)
end
function mnpgenerate(itersym1, itersym2, itersym3, returntypeexpr, prototype, bodyfunc, MNPlist)
    # Generate versions for specific dimensions
    fdim = [generate3(itersym1, itersym2, itersym3, prototype, bodyfunc, M, N, P) for (M,N,P) in MNPlist]
    
    # Generate the generic cache-based version
    fsym = funcsym(prototype)
    dictname = symbol(string(fsym)*"_cache")
    fargs = funcargs(prototype)
    flocal = funcrename(copy(prototype), :_F_)
    F = Expr(:function, prototype, quote
             if !haskey($dictname, ($itersym1, $itersym2, $itersym3))
                 gen3 = generate3($(symbol(itersym1)), $(symbol(itersym2)), $(symbol(itersym3)), $(Expr(:quote, flocal)), $bodyfunc, $itersym1, $itersym2, $itersym3)
                 $(dictname)[($itersym1, $itersym2, $itersym3)] = eval(quote
                     local _F_
                     $gen3
                     _F_
                 end)
             end
             ($(dictname)[($itersym1, $itersym2, $itersym3)]($(fargs...)))::$returntypeexpr
         end)
    Expr(:block, fdim..., quote
            let $dictname = Dict{(Int,Int,Int),Function}()
            $F
            end
        end)
end

isfuncexpr(ex::Expr) =
    ex.head == :function || (ex.head == :(=) && typeof(ex.args[1]) == Expr && ex.args[1].head == :call)
isfuncexpr(arg) = false

sreplace!(arg, sym, val) = arg
function sreplace!(ex::Expr, sym, val)
    for i = 1:length(ex.args)
        ex.args[i] = sreplace!(ex.args[i], sym, val)
    end
    ex
end
sreplace!(s::Symbol, sym, val) = s == sym ? val : s

intarith!(arg) = arg
function intarith!(ex::Expr)
    for i = 1:length(ex.args)
        ex.args[i]=intarith!(ex.args[i])
        if isa(ex.args[i],Expr) && ex.args[i].head==:call && (ex.args[i].args[1]==:+ || ex.args[i].args[1]==:- || ex.args[i].args[1]==:* || ex.args[i].args[1]==:div) && all(a->isa(a,Int),ex.args[i].args[2:end])
            ex.args[i]=eval(ex.args[i])
        end
    end
    ex
end

# Remove any function parameters that are integers
function spliceint!(ex::Expr)
    if ex.head == :escape
        return esc(spliceint!(ex.args[1]))
    end
    ex.head == :call || error(string(ex, " must be a call"))
    if isa(ex.args[1], Expr) && ex.args[1].head == :curly
        args = ex.args[1].args
        for i = length(args):-1:1
            if isa(args[i], Int)
                splice!(args, i)
            end
        end
    end
    ex
end

function popescape(ex::Expr)
    while ex.head == :escape
        ex = ex.args[1]
    end
    ex
end

# Extract the "function name"
function funcsym(prototype::Expr)
    prototype = popescape(prototype)
    prototype.head == :call || error(string(prototype, " must be a call"))
    tmp = prototype.args[1]
    if isa(tmp, Expr) && tmp.head == :curly
        tmp = tmp.args[1]
    end
    return tmp
end

function funcrename(prototype::Expr, name::Symbol)
    prototype = popescape(prototype)
    prototype.head == :call || error(string(prototype, " must be a call"))
    tmp = prototype.args[1]
    if isa(tmp, Expr) && tmp.head == :curly
        tmp.args[1] = name
    else
        prototype.args[1] = name
    end
    return prototype
end

function hasparameter(prototype::Expr, sym::Symbol)
    prototype = popescape(prototype)
    prototype.head == :call || error(string(prototype, " must be a call"))
    tmp = prototype.args[1]
    if isa(tmp, Expr) && tmp.head == :curly
        for i = 2:length(tmp.args)
            if tmp.args[i] == sym
                return true
            end
        end
    end
    false
end

# Extract the symbols of the function arguments
funcarg(s::Symbol) = s
funcarg(ex::Expr) = ex.args[1]
function funcargs(prototype::Expr)
    prototype = popescape(prototype)
    prototype.head == :call || error(string(prototype, " must be a call"))
    map(a->funcarg(a), prototype.args[2:end])
end

### Cartesian-specific macros

# Generate nested loops
macro nloops(N, itersym, rangeexpr, args...)
    _nloops(N, itersym, rangeexpr, args...)
end

_nloops(N::Int, itersym::Symbol, arraysym::Symbol, args::Expr...) = _nloops(N, itersym, :(d->1:size($arraysym,d)), args...)

function _nloops(N::Int, itersym::Symbol, rangeexpr::Expr, args::Expr...)
    if rangeexpr.head != :->
        error("Second argument must be an anonymous function expression to compute the range")
    end
    if !(1 <= length(args) <= 3)
        error("Too many arguments")
    end
    body = args[end]
    ex = Expr(:escape, body)
    for dim = 1:N
        itervar = inlineanonymous(itersym, dim)
        rng = inlineanonymous(rangeexpr, dim)
        preexpr = length(args) > 1 ? inlineanonymous(args[1], dim) : (:(nothing))
        postexpr = length(args) > 2 ? inlineanonymous(args[2], dim) : (:(nothing))
        ex = quote
            for $(esc(itervar)) = $(esc(rng))
                $(esc(preexpr))
                $ex
                $(esc(postexpr))
            end
        end
    end
    ex
end

macro stridedloops(N, itersym, dimsym, args...)
    _stridedloops(N, itersym, dimsym, args...)
end

function _stridedloops(N::Int, itersym::Symbol, dimsym::Symbol, args...)
    mod(length(args),3)==1 || error("Wrong number of arguments")
    body = args[end]
    ex = Expr(:escape, body)
    for i=1:3:length(args)-1
        ex=sreplace!(ex,args[i],inlineanonymous(args[i], 1))
    end
    for dim = 1:N
        itervar = inlineanonymous(itersym, dim)
        dimvar = inlineanonymous(dimsym, dim)
        preargs = {}
        postargs = {}
        for i=1:3:length(args)-1
            indnew = inlineanonymous(args[i], dim)
            start = (dim < N ? inlineanonymous(args[i], dim+1) : args[i+1])
            step = inlineanonymous(args[i+2], dim)
            push!(preargs,:($(esc(indnew)) = $(esc(start))))
            push!(postargs,:($(esc(indnew)) += $(esc(step))))
        end
        preexpr=Expr(:block,preargs...)
        postexpr=Expr(:block,postargs...)
        ex = quote
            $preexpr
            for $(esc(itervar)) = 1:$(esc(dimvar))
                $ex
                $postexpr
            end
        end
    end
    if N == 0 # even a zero-dimensional array has one element and thus requires a single run
        preargs = {}
        for i=1:3:length(args)-1
            indnew = inlineanonymous(args[i], 1)
            start = args[i+1]
            push!(preargs,:($(esc(indnew)) = $(esc(start))))
        end
        preexpr=Expr(:block,preargs...)
        ex = quote
            $preexpr
            $ex
        end
    end
    ex
end








# Generate expression A[i1, i2, ...]
macro nref(N, A, sym)
    _nref(N, A, sym)
end

function _nref(N::Int, A::Symbol, ex)
    vars = [ inlineanonymous(ex,i) for i = 1:N ]
    Expr(:escape, Expr(:ref, A, vars...))
end

# Generate f(arg1, arg2, ...)
macro ncall(N, f, sym...)
    _ncall(N, f, sym...)
end

function _ncall(N::Int, f, args...)
    pre = args[1:end-1]
    ex = args[end]
    vars = [ inlineanonymous(ex,i) for i = 1:N ]
    Expr(:escape, Expr(:call, f, pre..., vars...))
end

# Generate N expressions
macro nexprs(N, ex)
    _nexprs(N, ex)
end

function _nexprs(N::Int, ex::Expr)
    exs = [ inlineanonymous(ex,i) for i = 1:N ]
    Expr(:escape, Expr(:block, exs...))
end

# Make variables esym1, esym2, ... = isym
macro nextract(N, esym, isym)
    _nextract(N, esym, isym)
end

function _nextract(N::Int, esym::Symbol, isym::Symbol)
    aexprs = [Expr(:escape, Expr(:(=), inlineanonymous(esym, i), :(($isym)[$i]))) for i = 1:N]
    Expr(:block, aexprs...)
end

function _nextract(N::Int, esym::Symbol, ex::Expr)
    aexprs = [Expr(:escape, Expr(:(=), inlineanonymous(esym, i), inlineanonymous(ex,i))) for i = 1:N]
    Expr(:block, aexprs...)
end

# Check whether variables i1, i2, ... all satisfy criterion
macro nall(N, criterion)
    _nall(N, criterion)
end

function _nall(N::Int, criterion::Expr)
    if criterion.head != :->
        error("Second argument must be an anonymous function expression yielding the criterion")
    end
    conds = [Expr(:escape, inlineanonymous(criterion, i)) for i = 1:N]
    Expr(:&&, conds...)
end

macro ntuple(N, ex)
    _ntuple(N, ex)
end

function _ntuple(N::Int, ex)
    vars = [ inlineanonymous(ex,i) for i = 1:N ]
    Expr(:escape, Expr(:tuple, vars...))
end

# if condition1; operation1; elseif condition2; operation2; else operation3
# You can pass one or two operations; the second, if present, is used in the final "else"
macro nif(N, condition, operation...)
    # Handle the final "else"
    ex = esc(inlineanonymous(length(operation) > 1 ? operation[2] : operation[1], N))
    # Make the nested if statements
    for i = N-1:-1:1
        ex = Expr(:if, esc(inlineanonymous(condition,i)), esc(inlineanonymous(operation[1],i)), ex)
    end
    ex
end

## Utilities

# Simplify expressions like :(d->3:size(A,d)-3) given an explicit value for d
function inlineanonymous(ex::Expr, val)
    if ex.head != :->
        error("Not an anonymous function")
    end
    if !isa(ex.args[1], Symbol)
        error("Not a single-argument anonymous function")
    end
    sym = ex.args[1]
    ex = ex.args[2]
    exout = lreplace(ex, sym, val)
    exout = poplinenum(exout)
    exprresolve(exout)
end

# Given :i and 3, this generates :i_3
inlineanonymous(base::Symbol, ext) = symbol(string(base)*"_"*string(ext))

# Replace a symbol by a value or a "coded" symbol
# E.g., for d = 3,
#    lreplace(:d, :d, 3) -> 3
#    lreplace(:i_d, :d, 3) -> :i_3
#    lreplace(:i_{d-1}, :d, 3) -> :i_2
# This follows LaTeX notation.
lreplace(ex, sym::Symbol, val) = lreplace!(copy(ex), sym, val, Regex("_"*string(sym)*"(\$|(?=_))"))
lreplace!(arg, sym::Symbol, val, r) = arg
function lreplace!(s::Symbol, sym::Symbol, val, r::Regex)
    if (s == sym)
        return val
    end
    symbol(replace(string(s), r, "_"*string(val)))
end
function lreplace!(ex::Expr, sym::Symbol, val, r)
    # Curly-brace notation, which acts like parentheses
    if ex.head == :curly && length(ex.args) == 2 && isa(ex.args[1], Symbol) && endswith(string(ex.args[1]), "_")
        excurly = lreplace!(ex.args[2], sym, val, r)
        return symbol(string(ex.args[1])*string(exprresolve(excurly)))
    end
    for i in 1:length(ex.args)
        ex.args[i] = lreplace!(ex.args[i], sym, val, r)
    end
    ex
end

poplinenum(arg) = arg
function poplinenum(ex::Expr)
    if ex.head == :block
        if length(ex.args) == 1
            return ex.args[1]
        elseif length(ex.args) == 2 && ex.args[1].head == :line
            return ex.args[2]
        end
    end
    ex
end

exprresolve(arg) = arg
function exprresolve(ex::Expr)
    for i = 1:length(ex.args)
        ex.args[i] = exprresolve(ex.args[i])
    end
    # Handle simple arithmetic
    if ex.head == :call && in(ex.args[1], (:+, :-, :*, :/, :div)) && all([isa(ex.args[i], Number) for i = 2:length(ex.args)])
        return eval(ex)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-) && length(ex.args) == 3 && ex.args[3] == 0
        # simplify x+0 and x-0
        return ex.args[2]
    end
    # Resolve array references
    if ex.head == :ref && isa(ex.args[1], Array)
        for i = 2:length(ex.args)
            if !isa(ex.args[i], Real)
                return ex
            end
        end
        return ex.args[1][ex.args[2:end]...]
    end
    # Resolve conditionals
    if ex.head == :if
        try
            tf = eval(ex.args[1])
            ex = tf?ex.args[2]:ex.args[3]
        catch
        end
    end
    ex
end