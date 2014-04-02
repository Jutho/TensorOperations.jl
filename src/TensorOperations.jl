module TensorOperations

export @l_str
export tensorcopy, tensoradd, tensortrace, tensorcontract

# LabelError
#------------
# Tensor operations, either using methods or via index notation,
# are specified by assigning labels to the different indices of
# an array. All errors related to invalid label configurations
type LabelError <: Exception
    msg::String
end

# Tensor Operations
#-------------------
using Base.Cartesian
include("cache.jl")
include("blockdims.jl")
include("tensorcopy.jl")
include("tensoradd.jl")
include("tensortrace.jl")
include("tensorcontract.jl")


# Index Notation
#----------------
#include("indexnotation.jl")

end # module
