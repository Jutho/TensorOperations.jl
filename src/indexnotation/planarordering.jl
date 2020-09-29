# Functions for ordering contraction indices so as to avoid unnecessary braiding.
# TODO Maybe this should be a submodule? It would only need to export parsegraph and
# nextpair.

# A Node of the contraction graph is a Tuple of Any elements, that are the names of the
# edges connected to that Node, in counter clockwise order.
Node = Tuple
# An Edge is a Tuple of Symbol that are the names of Nodes that the Edge connects. The
# alternatively, the second element of the Tuple maybe be nothing, to indicate an open
# index.
Edge = Union{Tuple{Symbol, Symbol}, Tuple{Symbol, Nothing}}

# TODO I just made up the name RigidGraph, there must be an established math term for this.
"""
A RigidGraph is a graph for which the edges connected to a vertex come with an planar
ordering, counter clockwise around the vertex.
"""
struct RigidGraph
    # A Dict of edges by their names. The name of an Edge can be Any, but is usually a
    # Symbol or an Int, the same one used in the @tensor call.
    edges::Dict{Any, Edge}
    # A Dict of nodes by their names. The names of Nodes are Symbols that are automatically
    # generated.
    nodes::Dict{Symbol, Node}

    RigidGraph() = new(Dict{Any, Edge}(), Dict{Symbol, Node}())
end

function add_node!(g::RigidGraph, nodename::Symbol, node::Node)
    g.nodes[nodename] = node
    for edgename in node
        if edgename in keys(g.edges)
            e = g.edges[edgename]
            @assert e[2] == nothing
            g.edges[edgename] = (e[1], nodename)
        else
            g.edges[edgename] = (nodename, nothing)
        end
    end
    return g
end

"""
    nextpair(g::RigidGraph, edgename::Any, nodename::Symbol)

Given the names of an `Edge` and a `Node` of `g`, rotate counter clockwise around the `Node`
to its next edge. Return the name of this next edge, and either the name of the current
`Node`, if that edge is dangling (an open index), or the name of the `Node` at the other end
of the `Edge`, if the `Edge` is not dangling.
"""
function nextpair(g::RigidGraph, edgename::Any, nodename::Symbol)
    node = g.nodes[nodename]
    i = findfirst(en -> en == edgename, node)
    next_i = mod1(i+1, length(node))
    next_edgename = node[next_i]
    next_edge = g.edges[next_edgename]
    if next_edge[2] == nothing || next_edge[2] == nodename
        next_nodename = next_edge[1]
    else
        next_nodename = next_edge[2]
    end
    return next_edgename, next_nodename
end

"""
    parsegraph(ex::Expr)

Parse a tensor contraction expression into a `RigidGraph`.
"""
parsegraph(ex::Expr) = parsegraph!(RigidGraph(), ex)

function parsegraph!(g, ex::Expr)
    # TODO This should be able to handle a much wider range of Exprs, but for now I don't
    # need anything else, so this is for later.
    if ex.head == :call && ex.args[1] == :*
        g = parsegraph!(g, ex.args[2])
        g = parsegraph!(g, ex.args[3])
    elseif ex.head == :typed_vcat
        nodename = gensym()
        # The domain indices are listed in clockwise order, so reverse them.
        node = (parserow(ex.args[2])..., reverse(parserow(ex.args[3]))...)
        add_node!(g, nodename, node)
    elseif ex.head == :typed_hcat || ex.head == :ref
        nodename = gensym()
        node = (ex.args[2:end]...,)
        add_node!(g, nodename, node)
    else
        msg = "Unrecognized expression type in parsegraph, head = $(ex.head)"
        throw(ArgumentError(msg))
    end
    return g
end

# parserow deals with the fact that args[2] and args[3] of a typec_vcat expression may be a
# subexpression (several indices) or just an Int or a Symbol (a single index).
parserow(ex::Expr) = (@assert ex.head == :row, return ex.args)
parserow(x) = (x,)

"""
    planarsort(ex::Expr, leftinds::Vector{Any}, rightinds::Vector{Any})

Given a tensor contraction expression `ex` and `leftinds` and `rightinds` for it, return two
`Tuple`s that have the same indices as `leftinds` and `rightinds`, respectively, but where
the indices are reordered so that no braidings of the external indices are introduced.

This function is only guaranteed to work if the graph of `ex` is planar, and the indices in
`leftinds` and `rightinds` don't interleave. If this is not the case, usually the original
`leftinds` and `rightinds` are returned, though in some situations another permutation of
theirs may be returned.
"""
function planarsort(ex::Expr, leftinds::Vector{Any}, rightinds::Vector{Any})
    length(leftinds) == 0 && length(rightinds) == 0 && (return leftinds, rightinds)
    local g
    try
        g = parsegraph(ex)
    catch ArgumentError
        # TODO parsegraph throws an ArgumentError if it doesn't understand the expression.
        # Make it be able to deal with other types of expressions, and once that is done,
        # raise a warning here, instead of just silently returning.
        return leftinds, rightinds
    end
    # Starting from one of the external indices, go counter clockwise around the graph `g`,
    # following edges, using `nextpair`. Keep track of which (node, edge) pairs have been
    # visited, and stop once we return to our starting point. Build up, on the way,
    # `boundaryloop`, a vector of all the dangling edges that we meet, in the order that
    # we meet them.
    externalinds = Set{Any}(vcat(leftinds, rightinds))
    visited = Set{Any}()
    boundaryloop = Vector{Any}()
    local edgename
    if length(leftinds) > 0
        edgename = leftinds[1]
    elseif length(rightinds) > 0
        edgename = rightinds[end]
    end
    nodename = g.edges[edgename][1]
    while !((nodename, edgename) in visited)
        push!(visited, (nodename, edgename))
        edgename in externalinds && push!(boundaryloop, edgename)
        edgename, nodename = nextpair(g, edgename, nodename)
    end
    # Check that `boundaryloop` includes all the external indices. If not, just return the
    # original index lists, since a proper planar ordering isn't possible.
    if length(boundaryloop) == length(externalinds)
        if length(leftinds) > 0 && length(rightinds) > 0
            # `boundaryloop` includes all the external indices in a counter clockwise order
            # (assuming `g` is planar). The ordering is cyclic though, so rotate
            # `boundaryloop` so that the first index in it is in `leftinds`, and the last
            # one is in `rightinds`. Once this is done, if `leftinds` and `rightinds` don't
            # interleave, then `boundaryloop` should have all the indices in `leftinds`
            # first, followed by all the indices in `rightinds`.
            # TODO We do this permutation stupidly one step at a time. This is quite silly,
            # although pretty fool-proof.
            perm = vcat(2:length(boundaryloop), 1)
            while !(boundaryloop[1] in leftinds) || !(boundaryloop[end] in rightinds)
                boundaryloop = boundaryloop[perm]
            end
        end
        # Reorders left/rightinds to the order in which they appear in boundaryloop
        leftinds = intersect(boundaryloop, leftinds)
        rightinds = intersect(boundaryloop, rightinds)
    end
    return leftinds, rightinds
end
