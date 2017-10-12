module TensorOperations

import Base.Iterators.flatten
import Base.setindex

if VERSION < v"0.7.0-DEV.843"
    Base.@pure StaticLength(N) = Val{N}
else
    Base.@pure StaticLength(N) = Val{N}()
end

# 0.7.0-DEV.1993
@static if !isdefined(Base, :EqualTo)
    struct EqualTo{T} <: Function
        x::T
        EqualTo(x::T) where {T} = new{T}(x)
    end
    (f::EqualTo)(y) = isequal(f.x, y)
    const equalto = EqualTo
    export equalto
end


export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

export @tensor, @tensoropt, @optimalcontractiontree

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
include("indexnotation/nconstyle.jl")
include("indexnotation/poly.jl")
include("indexnotation/optimize.jl")
include("indexnotation/indexedobject.jl")
include("indexnotation/sum.jl")
include("indexnotation/product.jl")

# Functions
#----------
include("functions/simple.jl")
include("functions/inplace.jl")


end # module
