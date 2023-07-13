const defaultparser = TensorParser()

"""
    @notensor(block)

Marks a block which should be ignored within an `@tensor` environment. Has no effect outside of `@tensor`.
"""
macro notensor(ex::Expr)
    return esc(ex)
end

"""
    @tensor(tensor_expr; kwargs...)
    @tensor [kw_expr] tensor_expr

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, this will be evaluated as pairwise contractions in left to right
order, unless the so-called NCON style is used (positive integers for contracted
indices and negative indices for open indices).

Additional keyword arguments may be passed to control the behavior of the parser:

- `order`: 
    A list of contraction indices of the form `order=(...,)` which specify the order in which they will be contracted.
- `opt`:
    Contraction order optimization, similar to [`@tensoropt`](@ref). Can be either a boolean or an `OptExpr`.
- `contractcheck`:
    Boolean flag to enable runtime check for contractibility of indices with clearer error messages.
- `costcheck`:
    Adds runtime checks to ensure that the contraction order is optimal. Can be either `:warn` or `:cache`. The former will issues warnings when sub-optimal expressions are encountered, while the latter will cache the optimal contraction order for each tensor site and calling site.
- `backend`: 
    Inserts a backend call for the different tensor operations.
"""
macro tensor(ex::Expr)
    return esc(defaultparser(ex))
end

function standardize_kwargs(ex)
    ex.head == :parameters && return ex
    ex.head == :(=) && return Expr(:parameters, Expr(:kw, ex.args...))
    if ex.head == :tuple
        params = map(ex.args) do x
            return Expr(:kw, x.args...)
        end
        return Expr(:parameters, params...)
    end
    throw(ArgumentError("unknown keyword expression `$ex`"))
end

macro tensor(kwargsex::Expr, ex::Expr)
    kwargs = standardize_kwargs(kwargsex)
    parser = TensorParser()

    for param in kwargs.args
        name, val = param.args

        if name == :order
            (val isa Expr && val.head == :tuple) ||
                throw(ArgumentError("Invalid use of `order`, should be `order=(...,)`"))
            indexorder = map(normalizeindex, val.args)
            parser.contractiontreebuilder = network -> indexordertree(network, indexorder)

        elseif name == :contractcheck
            val isa Bool ||
                throw(ArgumentError("Invalid use of `contractcheck`, should be `contractcheck=bool`."))
            val && push!(parser.preprocessors, ex -> insertcontractionchecks(ex))

        elseif name == :costcheck
            val in (:warn, :cache) ||
                throw(ArgumentError("Invalid use of `costcheck`, should be `costcheck=warn` or `costcheck=cache`"))
            parser.contractioncostcheck = val
        elseif name == :opt
            if val isa Bool && val
                optdict = optdata(ex)
            elseif val isa Expr
                optdict = optdata(val, ex)
            else
                throw(ArgumentError("Invalid use of `opt`, should be `opt=true` or `opt=OptExpr`"))
            end
            parser.contractiontreebuilder = network -> optimaltree(network, optdict)[1]
        elseif name == :backend
            val isa Symbol ||
                throw(ArgumentError("Backend should be a symbol."))
            backend = val
            push!(parser.postprocessors, ex -> insertbackend(ex, backend))
        else
            throw(ArgumentError("Unknown keyword argument `name`."))
        end
    end

    return esc(parser(ex))
end

"""
    @tensoropt(optex, block)
    @tensoropt(block)

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, the macro will determine (at compile time) the optimal contraction
order depending on the cost associated to the individual indices. If no `optex` is
provided, all indices are assumed to have an abstract scaling `χ` which is optimized
in the asympotic limit of large `χ`.

The cost can be specified in the following ways:

```julia
@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
@tensoropt (a,b,c,e) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost χ for indices a,b,c,e, other indices (d,f) have cost 1
@tensoropt !(a,b,c,e) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# cost 1 for indices a,b,c,e; other indices (d,f) have asymptotic cost χ
@tensoropt C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost χ for all indices (a,b,c,d,e,f)
```

Note that `@tensoropt` will optimize any tensor contraction sequence it encounters
in the (block of) expressions. It will however not break apart expressions that have
been explicitly grouped with parenthesis, i.e. in
```julia
@tensoroptC[a,b,c,d] := A[a,e,c,f,h]*(B[f,g,e,b]*C[g,d,h])
```
it will always contract `B` and `C` first. For a single tensor contraction sequence,
the optimal contraction order and associated (asymptotic) cost can be obtained using
`@optimalcontractiontree`.
"""
macro tensoropt(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    parser = TensorParser()
    parser.contractiontreebuilder = network -> optimaltree(network, optdict)[1]
    return esc(parser(ex))
end

"""
    @tensoropt_verbose(optex, block)
    @tensoropt_verbose(block)

Similar to `@tensoropt`, but prints information details regarding the optimization process to the standard output.
"""
macro tensoropt_verbose(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    parser = TensorParser()
    parser.contractiontreebuilder = network -> optimaltree(network, optdict; verbose=true)[1]
    return esc(parser(ex))
end

macro optimalcontractiontree(expressions...)
    if length(expressions) == 1
        ex = expressions[1]
        optdict = optdata(ex)
    elseif length(expressions) == 2
        optex = expressions[1]
        ex = expressions[2]
        optdict = optdata(optex, ex)
    end

    if isassignment(ex) || isdefinition(ex)
        ex = getrhs(ex)
    end
    if !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k in 2:length(ex.args)]
    tree, cost = optimaltree(network, optdict)
    return tree, cost
end

"""
    @ncon(tensorlist, indexlist; order = ..., output = ...)

Contract the tensors in `tensorlist` (of type `Vector` or `Tuple`) according to the network
as specified by `indexlist`. Here, `indexlist` is a list (i.e. a `Vector` or `Tuple`) with
the same length as `tensorlist` whose entries are themselves lists (preferably
`Vector{Int}`) where every integer entry provides a label for corresponding index/dimension
of the corresponding tensor in `tensorlist`. Positive integers are used to label indices
that need to be contracted, and such thus appear in two different entries within
`indexlist`, whereas negative integers are used to label indices of the output tensor, and
should appear only once.

By default, contractions are performed in the order such that the indices being contracted
over are labelled by increasing integers, i.e. first the contraction corresponding to label
`1` is performed. The output tensor had an index order corresponding to decreasing
(negative, so increasing in absolute value) index labels. The keyword arguments `order` and
`output` allow to change these defaults.

The advantage of the macro `@ncon` over the function call `ncon` is that the former
automatically generates a unique symbol that hooks into the cache. Furthermore, if
`tensorlist` is not just some variable but an actual list (as a tuple with parentheses or a
vector with square brackets) at the call site, the `@ncon` macro will scan for conjugation
calls, e.g. `conj(A)`, and replace this with just `A` but build a matching list of
conjugation flags to be specified to `ncon`. This makes it more convenient to specify
tensor conjugation, without paying the cost of actively performing the conjugation
beforehand.

See also the function [`ncon`](@ref).
"""
macro ncon(args...)
    if length(args) == 2
        return _nconmacro(args[1], args[2])
    else
        return _nconmacro(args[2], args[3], args[1])
    end
end
function _nconmacro(tensors, indices, kwargs=nothing)
    if !(tensors isa Expr) # there is not much that we can do
        if kwargs === nothing
            ex = Expr(:call, :ncon, tensors, indices,
                      Expr(:call, :fill, false, Expr(:call, :length, tensors)))
        else
            ex = Expr(:call, :ncon, kwargs, tensors, indices,
                      Expr(:call, :fill, false, Expr(:call, :length, tensors)))
        end
        return esc(ex)
    end
    if tensors.head == :vect || tensors.head == :tuple
        tensorargs = tensors.args
    elseif tensors.head == :ref
        tensorargs = tensors.args[2:end]
    else
        throw(ArgumentError("invalid @ncon syntax"))
    end
    if any(isa(ta, Expr) && ta.head === :... for ta in tensorargs)
        throw(ArgumentError("@ncon does not support splats (...) in tensor lists."))
    end
    conjlist = fill(false, length(tensorargs))
    for i in 1:length(tensorargs)
        if tensorargs[i] isa Expr
            if tensorargs[i].head == :call && tensorargs[i].args[1] == :conj
                tensorargs[i] = tensorargs[i].args[2]
                conjlist[i] = true
            end
        end
    end
    if tensors.head == :ref
        tensorex = Expr(:ref, tensors.args[1], tensorargs...)
    else
        tensorex = Expr(:ref, :Any, tensorargs...)
    end
    if kwargs === nothing
        ex = Expr(:call, :ncon, tensorex, indices, conjlist)
    else
        ex = Expr(:call, :ncon, kwargs, tensorex, indices, conjlist)
    end
    return esc(ex)
end

"""
    @cutensor tensor_expr

Use the GPU to perform all tensor operations, through the use of the cuTENSOR library.
This will transfer all arrays to the GPU before performing the requested operations. If the
output is an existing host array, the result will be transferred back. If a new array is
created (i.e. using `:=`), it will remain on the GPU device and it is up to the user to
transfer it back. This macro is equivalent to `@tensor backend=cuTENSOR tensor_expr`.

!!! note
    This macro requires the cuTENSOR library to be installed and loaded. This can be
    achieved by running `using cuTENSOR` or `import cuTENSOR` before using the macro.
"""
macro cutensor(ex::Expr)
    haskey(Base.loaded_modules, Base.identify_package("cuTENSOR")) ||
        throw(ArgumentError("cuTENSOR not loaded: add `using cuTENSOR` or `import cuTENSOR` before using `@cutensor`"))
    parser = TensorParser()
    push!(parser.postprocessors, ex -> insertbackend(ex, :cuTENSOR))
    return esc(parser(ex))
end
