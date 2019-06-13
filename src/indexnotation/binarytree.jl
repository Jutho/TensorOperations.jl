struct BinaryTreeNode
    left
    right
end

Base.show(io::IO, blk::BinaryTreeNode) = show(io, "plain/text", blk)
function Base.show(io::IO, ::MIME"plain/text", blk::BinaryTreeNode)
    print(io, "Contraction Tree: ")
    print_tree(io, blk)
end

function print_tree(io::IO, blk::BinaryTreeNode, print_level=0)
    print(io, "(")
    print_tree(io, blk.left, print_level+1)
    print(io, " ↔ ")
    print_tree(io, blk.right, print_level+1)
    print(io, ")")
end

function print_tree(io::IO, blk, print_level=0)
    print(io, blk)
end

Base.getindex(t::BinaryTreeNode, i::Int) = i==1 ? t.left : (i==2 ? t.right : throw(BoundsError(t, i)))
Base.iterate(t::BinaryTreeNode, args...) = iterate((t.left, t.right), args...)
