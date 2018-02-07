__precompile__(true)
module TensorOperations

import Base.Iterators.flatten
import Base.setindex

if VERSION < v"0.7-"
    Base.@pure StaticLength(N) = Val{N}

    using Base.LinAlg.BlasFloat
    const Nothing = Void
else
    Base.@pure StaticLength(N) = Val{N}()

    using LinearAlgebra
    using LinearAlgebra.BLAS
    using LinearAlgebra: BlasFloat
end
_findfirst(args...) = (i = findfirst(args...); isa(i, Nothing) ? 0 : i)
_findnext(args...) = (i = findnext(args...); isa(i, Nothing) ? 0 : i)
_findlast(args...) = (i = findlast(args...); isa(i, Nothing) ? 0 : i)


using Compat
using Compat: copyto!

export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

export @tensor, @tensoropt, @optimalcontractiontree

const IndexTuple{N} = NTuple{N,Int}

# Auxiliary functions
#---------------------
include("auxiliary/axpby.jl")
include("auxiliary/error.jl")
include("auxiliary/meta.jl")
include("auxiliary/stridedarray.jl")
include("auxiliary/strideddata.jl")
include("auxiliary/unique2.jl")

# Implementations
#-----------------
include("implementation/indices.jl")
include("implementation/kernels.jl")
include("implementation/recursive.jl")
include("implementation/stridedarray.jl")
include("implementation/strides.jl")


# Index notation
#----------------
include("indexnotation/tensormacro.jl")
include("indexnotation/tensorexpressions.jl")
include("indexnotation/ncontree.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/poly.jl")

# Functions
#----------
include("functions/simple.jl")
include("functions/inplace.jl")

precompile(tensorify, (Expr,))
precompile(optdata,(Expr,))
precompile(optdata,(Expr,Expr))
#
end # module
