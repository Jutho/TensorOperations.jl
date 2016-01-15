# auxiliary/error.jl
#
# An exception type for reporting errors in the index specificatino

immutable IndexError{S<:AbstractString} <: Exception
    msg::S
end

checkindices(A, IA) =
    length(IA) == numind(A) || throw(IndexError("invalid number of indices: $IA"))
