module TensorOperations

export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

export @tensor

# Auxiliary functions
#---------------------
include("aux/axpby.jl")
include("aux/error.jl")
include("aux/meta.jl")
include("aux/stridedarray.jl")
include("aux/strideddata.jl")
include("aux/unique2.jl")

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
include("indexnotation/indexedobject.jl")
include("indexnotation/sum.jl")
include("indexnotation/product.jl")

# Functions
#----------
include("functions/simple.jl")
include("functions/inplace.jl")


end # module
