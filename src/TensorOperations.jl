module TensorOperations

export @l_str, LabelError
export tensorcopy, tensoradd, tensortrace, tensorcontract, scalar

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
const OBASELENGTH=16 # square root of total size of all open dimensions
const CBASELENGTH=24 # total size of all contraction dimensions
const PERMUTEBASELENGTH=1024


# Tensor Operations
#-------------------
include("cartesian.jl")
include("tensorcopy.jl")
include("tensoradd.jl")
# include("tensortrace.jl")
include("tensorcontract.jl")

# Scalar
#--------
scalar{T}(C::StridedArray{T,0})=C[1]
# scalar{T}(C::LabeledArray{T,0})=C.data[1]


# # Index Notation
# #----------------
# include("indexnotation.jl")

end # module
