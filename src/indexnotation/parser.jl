mutable struct TensorParser
    preprocessors::Vector{Any} # any preprocessing steps
    contractiontreebuilder::Any # determine a contraction tree for a contraction involving multiple tensors
    contractiontreesorter::Any # transforms the contraction expression into an expression of nested binary contractions using the tree output from the contractiontreebuilder

    postprocessors::Vector{Any}
    function TensorParser()
        preprocessors = [ex->replaceindices(normalizeindex, ex),
                            expandconj,
                            nconindexcompletion,
                            extracttensorobjects]
        contractiontreebuilder = defaulttreebuilder
        contractiontreesorter = defaulttreesorter
        postprocessors = [_flatten, removelinenumbernode, addtensoroperations]
        return new(preprocessors,
                    contractiontreebuilder,
                    contractiontreesorter,
                    postprocessors)
    end
end

function (parser::TensorParser)(ex)
    for p in parser.preprocessors
        ex = p(ex)
    end
    treebuilder = parser.contractiontreebuilder
    treesorter = parser.contractiontreesorter
    ex = processcontractions(ex, treebuilder, treesorter)
    ex = tensorify(ex)
    for p in parser.postprocessors
        ex = p(ex)
    end
    return ex
end

function processcontractions(ex::Expr, treebuilder, treesorter)
    ex = Expr(ex.head, map(e->processcontractions(e, treebuilder, treesorter), ex.args)...)
    if istensorcontraction(ex) && length(ex.args) > 3
        args = ex.args[2:end]
        network = map(getindices, args)
        for a in getallindices(ex)
            count(a in n for n in network) <= 2 ||
                throw(ArgumentError("invalid tensor contraction: $ex"))
        end
        tree = treebuilder(network)
        ex = treesorter(args, tree)
    end
    return ex
end
processcontractions(ex, treebuilder, treesorter) = ex

function defaulttreesorter(args, tree)
    if isa(tree, Int)
        return args[tree]
    else
        return Expr(:call, :*,
                        defaulttreesorter(args, tree[1]),
                        defaulttreesorter(args, tree[2]))
    end
end

function defaulttreebuilder(network)
    if isnconstyle(network)
        tree = ncontree(network)
    else
        tree = Any[1,2]
        for k = 3:length(network)
            tree = Any[tree, k]
        end
    end
    return tree
end

# functions for parsing and processing tensor expressions
function tensorify(ex::Expr)
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :throw
            return rhs
        end

        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)
            if hastraceindices(lhs)
                err = "left hand side of an assignment should have unique indices: $lhs"
                return :(throw(IndexError($err)))
            end
            dst, leftind, rightind = decomposetensor(lhs)
            if Set(vcat(leftind,rightind)) != Set(indices)
                err = "non-matching indices between left and right hand side: $ex"
                return :(throw(IndexError($err)))
            end
            if isassignment(ex)
                if ex.head == :(=)
                    return instantiate(dst, false, rhs, true, leftind, rightind)
                elseif ex.head == :(+=)
                    return instantiate(dst, true, rhs, 1, leftind, rightind)
                else
                    return instantiate(dst, true, rhs, -1, leftind, rightind)
                end
            else
                return Expr(:(=), dst, instantiate(nothing, false, rhs, true, leftind, rightind, false))
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                return Expr(ex.head, instantiate_scalar(lhs), Expr(:call, :scalar, instantiate(nothing, false, rhs, true, [], [], true)))
            elseif isscalarexpr(rhs)
                return Expr(ex.head, instantiate_scalar(lhs), instantiate_scalar(rhs))
            end
        else
            return ex # likely an error
        end
    end
    if ex.head == :block
        return Expr(ex.head, map(tensorify, ex.args)...)
    end
    if ex.head == :for
        return Expr(ex.head, ex.args[1], tensorify(ex.args[2]))
    end
    # constructions of the form: a = @tensor ...
    if isscalarexpr(ex)
        return instantiate_scalar(ex)
    end
    if istensorexpr(ex)
        if !isempty(getindices(ex))
            err = "cannot evaluate $ex to a scalar: uncontracted indices"
            return :(throw(IndexError($err)))
        end
        return Expr(:call, :scalar, instantiate(nothing, false, ex, true, [], [], true))
    end
    error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex) = ex

#
#
#
# #
# #
# #
# #
# #
# #         β =
# #         out = Expr(:=, dst, Expr(:call, :instantiate, tensorify(getrhs(ex))),)
# #         return esc(:($dst = instantiate($rhs, ))
# #     if ex.head == :(=) || ex.head == :(:=) || ex.head == :(+=) || ex.head == :(-=)
# #         lhs = ex.args[1]
# #         rhs = ex.args[2]
# #         if isa(lhs, Expr) && lhs.head == :ref
# #             dst = tensorify(lhs.args[1])
# #             src = ex.head == :(-=) ? tensorify(Expr(:call,:-,rhs)) : tensorify(rhs)
# #             indices = makeindex_expr(lhs)
# #             if ex.head == :(:=)
# #                 return :($dst = instantiate($src, $indices))
# #             else
# #                 value = ex.head == :(=) ? 0 : +1
# #                 return :(instantiate!($dst, $src, $indices, $value))
# #             end
# #         end
# #     end
# #     if ex.head == :ref
# #         indices = makeindex_expr(ex)
# #         t = tensorify(ex.args[1])
# #         return :(indexify($t,$indices))
# #     end
# #     if ex.head == :call && ex.args[1] == :scalar
# #         if length(ex.args) != 2
# #             error("scalar accepts only a single argument")
# #         end
# #         src = tensorify(ex.args[2])
# #         indices = :(Indices{()}())
# #         return :(scalar(instantiate($src, $indices)))
# #     end
# #     return Expr(ex.head,map(tensorify,ex.args)...)
# # end
# # tensorify(ex::Symbol) = esc(ex)
# # tensorify(ex) = ex
#
# function makeindex_expr(ex::Expr)
#     if ex.head == :ref
#         for i = 2:length(ex.args)
#             isa(ex.args[i],Int) || isa(ex.args[i],Symbol) || isa(ex.args[i],Char) || error("cannot make indices from $ex")
#         end
#     else
#         error("cannot make indices from $ex")
#     end
#     return :(Indices{$(tuple(ex.args[2:end]...))}())
# end
#
#
# function canonicalindex(ex)
#     if isa(ex, Symbol) || isa(ex, Int)
#         return ex
#     elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
#         return Symbol(canonicalindex(ex.args[1]), "′")
#     else
#         throw(ArgumentError("not a valid index: $ex"))
#     end
# end
# function quoteindex(ex)
#     if isa(ex, Symbol)
#         return QuoteNode(ex)
#     elseif isa(ex, Int)
#         return ex
#     else
#         throw(ArgumentError("not a valid index: $ex"))
#     end
# end
#
# # processcontractorder: convert multi-argument multiplication into tree of pairwise multiplication
# function processcontractorder(ex::Expr, optdata)
#     ex = Expr(ex.head, map(e->processcontractorder(e, optdata), ex.args)...)
#     if ex.head == :call && ex.args[1] == :* && length(ex.args) > 3
#         args = ex.args[2:end]
#         network = map(getindices, args)
#         err = "invalid tensor contraction: $ex"
#         for a in getallindices(ex)
#             count(a in n for n in network) <= 2 || return :(throw(IndexError($err)))
#         end
#         if optdata === nothing
#             if isnconstyle(network)
#                 tree = ncontree(network)
#                 ex = tree2expr(args, tree)
#             else
#                 ex = Expr(:call, :*, args[1], args[2])
#                 for k = 3:length(args)
#                     ex = Expr(:call, :*, ex, args[k])
#                 end
#             end
#         else
#             tree, = optimaltree(network, optdata)
#             ex = tree2expr(args, tree)
#         end
#     end
#     return ex
# end
# processcontractorder(ex, optdata) = ex
