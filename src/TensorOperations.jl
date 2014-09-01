module TensorOperations

export LabelError
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar, reset_tcbuffer

# LabelError
#------------
# Tensor operations, either using methods or via index notation,
# are specified by assigning labels to the different indices of
# an array. All errors related to invalid label configurations
type LabelError <: Exception
    msg::String
end

# Constants that define base case in recursive algorithms
#---------------------------------------------------------
# for tensorcopy, tensoradd and tensortrace
const TBASELENGTH=512
# note: total number elements involved = 2*512 = 1024

# for tensorcontract
const OBASELENGTH=16 # total size of all open dimensions in one of the two contraction partners
const CBASELENGTH=24 # total size of all contraction dimensions
# note: total number elements involved = 16*24*2+16*16 = 1024

# Tensor Operations
#-------------------
using Cartesian

include("kernels.jl")
include("tensorcopy.jl")
include("tensoradd.jl")
include("tensortrace.jl")
include("tensorcontract.jl")
include("tensorproduct.jl")

# Scalar
#--------
scalar(C::StridedArray)=length(C)==1 ? C[1] : throw(DimensionMismatch())

end # module

# Index Notation
#----------------
include("indexnotation.jl")
