module TensorOperations

export @l_str
export tensorcopy, tensoradd, tensortrace, tensorcontract

using Base.Cartesian
include("cache.jl")
include("blockdims.jl")
include("tensorcopy.jl")
include("tensoradd.jl")
# include("tensortrace.jl")
include("tensorcontract.jl")

# LabelList
#-------------
# Wrapper for a list of labels in the form of a Vector{Symbol}, so
# that we do not have to overload the getindex and setindex! methods
# of Array on a generic Vector{Symbol} argument, which might conflict
# with other packages or user definitions...

type LabelList
    labels::Vector{Symbol}
end
Base.length(l::LabelList)=length(l.labels)
LabelList(s::String)=LabelList(map(symbol,map(strip,split(s,','))))
macro l_str(s)
    LabelList(s)
end

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
LabeledArray{T,N}(A::StridedArray{T,N},l::LabelList)=LabeledArray{T,N}(A,l)

Base.eltype{T}(::LabeledArray{T})=T
Base.eltype{T}(::LabeledArray{T})=T
Base.eltype{T}(::Type{LabeledArray{T}})=T
Base.eltype{T,N}(::Type{LabeledArray{T,N}})=T

Base.getindex(A::Array,labels::LabelList)=LabeledArray(A,labels)
Base.getindex(A::SubArray,labels::LabelList)=LabeledArray(A,labels)
Base.getindex(A::SharedArray,labels::LabelList)=LabeledArray(A,labels)
Base.getindex(A::LabeledArray,labels::LabelList)=LabeledArray(A.data,labels)

Base.setindex!(A::Array,B::LabeledArray,l::LabelList)=tensorcopy!(A,l.labels,B.data,B.labels)
Base.setindex!(A::SubArray,B::LabeledArray,l::LabelList)=tensorcopy!(A,l.labels,B.data,B.labels)
Base.setindex!(A::SharedArray,B::LabeledArray,l::LabelList)=tensorcopy!(A,l.labels,B.data,B.labels)

# addition of arrays
+(A::LabeledArray,B::LabeledArray)=tensoradd(A.data,A.labels,B.data,B.labels)

# multiplication with scalars
Base.scale(A::LabeledArray,a::Number)=LabeledArray(scale(A.data,a),A.labels)
*(t::LabeledArray,a::Number)=scale(t,a)
*(a::Number,t::LabeledArray)=scale(t,a)
/(t::LabeledArray,a::Number)=scale(t,one(a)/a)
\(a::Number,t::LabeledArray)=scale(t,one(a)/a)

# general contraction
*(A::LabeledArray,B::LabeledArray)=tensorcontract(A.data,A.labels,B.data,B.labels)

end # module
