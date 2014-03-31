module TensorOperations

export tensorcopy, tensorcopy!
export tensoradd, tensoradd!
export tensordot
export tensorproject, tensorproject!
export tensortrace, tensortrace!
export tensorcontract, tensorcontract!

using Base.Cartesian
include("cache.jl")
include("level1.jl")
#include("level2.jl")
include("level3.jl")


# LabelList
#-------------
# Wrapper for a list of labels in the form of a Vector{Symbol}, so
# that we do not have to overload the getindex and setindex! methods
# of Array on a generic Vector{Symbol} argument, which might conflict
# with other packages or user definitions...

type LabelList
    labels::Vector{Symbol}
end
Base.length(l::IndexLabels)=length(l.labels)
IndexLabels(s::String)=IndexLabels(map(symbol,map(strip(split(s,',')))))
macro l_str(s)=IndexLabels(s)

type LabelError <: Exception
    msg::String
end


# LabeledArray
#--------------
# Wraps an Array with a LabelList. This type acts as return type
# of getindex(::Array,::LabelList) and can engage in tensor operations.

type LabeledArray{T,N}
    data::StridedArray{T,N}
    labels::Vector{Symbol}
    function LabeledArray(data::StridedArray{T,N},l::LabelList)
        if length(l)!=N
            throw(LabelError("Provide one label per index"))
        end
        new(data,l.labels)
    end
end
function LabeledArray{T,N}(A::StridedArray{T,N},l::LabelList)=LabeledArray{T,N}(A,l)

eltype{T}(::LabeledArray{T})=T
eltype{T}(::LabeledArray{T})=T
eltype{T}(::Type{LabeledArray{T}})=T
eltype{T,N}(::Type{LabeledArray{T,N}})=T

Base.getindex(A::StridedArray,labels::LabelList)=LabeledArray(A,labels)
Base.getindex(A::LabeledArray,labels::LabelList)=LabeledArray(A.data,labels)

Base.setindex(A::StridedArray,B::LabeledArray,l::LabelList)=tensorcopy!(A,l.labels,B.data,B.labels)

# addition of arrays
+(A::LabeledArray,B::LabeledArray)=tensoradd(A.data,A.labels,B.data,B.labels)

# multiplication with scalars
scale(A::LabeledArray,a::Number)=LabeledArray(scale(A.data,a),A.labels)
*(t::LabeledArray,a::Number)=scale(t,a)
*(a::Number,t::LabeledArray)=scale(t,a)
/(t::LabeledArray,a::Number)=scale(t,one(a)/a)
\(a::Number,t::LabeledArray)=scale(t,one(a)/a)

# general contraction
*(A::LabeledArray,B::LabeledArray)=tensorcontract(A.data,A.labels,B.data,B.labels)

end # module
