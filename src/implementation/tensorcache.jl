similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol) =
    checked_similar_from_indices(nothing, T, p1, p2, A, CA)

function cached_similar_from_indices(sym::Symbol, T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol)
    if use_cache()
        key = (sym, Threads.threadid())
        C = get(cache, key, nothing)
        C′ = checked_similar_from_indices(C, T, p1, p2, A, CA)
        cache[key] = C′
        return C′
    else
        return checked_similar_from_indices(nothing, T, p1, p2, A, CA)
    end
end

similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple, p1::IndexTuple, p2::IndexTuple,
    A, B, CA::Symbol, CB::Symbol) = checked_similar_from_indices(nothing, T, poA, poB, p1, p2, A, B, CA, CB)

function cached_similar_from_indices(sym::Symbol, T::Type, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol)

    if use_cache()
        key = (sym, Threads.threadid())
        C = get(cache, key, nothing)
        C′ = checked_similar_from_indices(C, T, poA, poB, p1, p2, A, B, CA, CB)
        cache[key] = C′
        return C′
    else
        return checked_similar_from_indices(nothing, T, poA, poB, p1, p2, A, B, CA, CB)
    end
end
