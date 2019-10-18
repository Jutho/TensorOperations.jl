const defaultparser = TensorParser()

"""
<<<<<<< HEAD
    @notensor(block)

Marks a block which should be ignored within an `@tensor` environment. Has no effect outside of `@tensor`.
"""
macro notensor(ex::Expr)
    return esc(ex)
end

"""
    @tensor(block)
=======
    @tensor(block [, order = (...)])
>>>>>>> eae44d1... add optional contraction order and corresponding treebuilder

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, this will be evaluated as pairwise contractions in left to right
order, unless the so-called NCON style is used (positive integers for contracted
indices and negative indices for open indices).

A second argument to the `@tensor` macro can be provided of the form `order=(...)`, where
the list specifies the contraction indices in the order in which they will be contracted.
"""
macro tensor(ex::Expr)
    return esc(defaultparser(ex))
end

macro tensor(ex::Expr, orderex::Expr)
    parser = TensorParser()
    if !(orderex.head == :(=) && orderex.args[1] == :order &&
            orderex.args[2] isa Expr && orderex.args[2].head == :tuple)
        throw(ArgumentError("unkown first argument in @tensor, should be `order = (...,)`"))
    end
    indexorder = map(normalizeindex, orderex.args[2].args)
    parser = TensorParser()
    parser.contractiontreebuilder = network->indexordertree(network, indexorder)
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
    parser.contractiontreebuilder = network->optimaltree(network, optdict)[1]
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
    parser.contractiontreebuilder = network->optimaltree(network, optdict; verbose = true)[1]
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
    network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
    tree, cost = optimaltree(network, optdict)
    return tree, cost
end
