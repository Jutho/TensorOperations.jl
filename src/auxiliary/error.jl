# auxiliary/error.jl
#
# An exception type for reporting errors in the index specificatino

struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

"""
    checkindices(A, IA)

Checks whether the indices in `IA` are compatible with the object `A`. The fallback
method checks `length(IA) == numind(A)`.
"""
checkindices(A, IA) =
    length(IA) == numind(A) || throw(IndexError("invalid number of indices: $IA"))
