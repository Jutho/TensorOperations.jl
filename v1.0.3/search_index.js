var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "TensorOperations.jl",
    "title": "TensorOperations.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#TensorOperations.jl-1",
    "page": "TensorOperations.jl",
    "title": "TensorOperations.jl",
    "category": "section",
    "text": "Fast tensor operations using a convenient Einstein index notation."
},

{
    "location": "#Table-of-contents-1",
    "page": "TensorOperations.jl",
    "title": "Table of contents",
    "category": "section",
    "text": "Pages = [\"index.md\", \"indexnotation.md\", \"functions.md\", \"cache.md\", \"implementation.md\"]\nDepth = 4"
},

{
    "location": "#Installation-1",
    "page": "TensorOperations.jl",
    "title": "Installation",
    "category": "section",
    "text": "Install with the package manager, pkg> add TensorOperations."
},

{
    "location": "#Package-features-1",
    "page": "TensorOperations.jl",
    "title": "Package features",
    "category": "section",
    "text": "A macro @tensor for conveniently specifying tensor contractions and index permutations   via Einstein\'s index notation convention. The index notation is analyzed at compile time.\nAbility to   optimize pairwise contraction order   using the @tensoropt macro.\nSupport for any Julia Base array which qualifies as strided, i.e. such that its entries   are layed out according to a regular pattern in memory. The only exception are   ReinterpretedArray objects (implementation provided by Strided.jl, see below).   Additionally, Diagonal objects whose underlying diagonal data is stored as a strided   vector are supported. This facilitates tensor contractions where one of the operands is   e.g. a diagonal matrix of singular values or eigenvalues, which are returned as a   Vector by Julia\'s eigen or svd method.\nImplementation can easily be extended to other types, by overloading a small set of   methods.\nEfficient implementation of a number of basic tensor operations (see below), by relying   on Strided.jl and gemm from BLAS for   contractions. The latter is optional but on by default, it can be controlled by a   package wide setting via enable_blas() and disable_blas(). If BLAS is disabled or   cannot be applied (e.g. non-matching or non-standard numerical types), Strided.jl is   also used for the contraction.\nA package wide cache for storing temporary arrays that are generated when evaluating   complex tensor expressions within the @tensor macro (based on the implementation of   LRUCache). By default, the cache is   allowed to use up to 50% of the total machine\'s memory."
},

{
    "location": "#Tensor-operations-1",
    "page": "TensorOperations.jl",
    "title": "Tensor operations",
    "category": "section",
    "text": "TensorOperations.jl is centered around 3 basic tensor operations, i.e. primitives in which every more complicated tensor expression is deconstructed.addition: Add a (possibly scaled version of) one array to another array, where the  indices of the both arrays might appear in different orders. This operation combines  normal array addition and index permutation. It includes as a special case copying one  array into another with permuted indices.\nThe actual implementation is provided by Strided.jl, which contains multithreaded implementations and cache-friendly blocking  strategies for an optimal efficiency.\ntrace or inner contraction: Perform a trace/contraction over pairs of indices of an  array, where the result is a lower-dimensional array. As before, the actual  implementation is provided by Strided.jl.\ncontraction: Performs a general contraction of two tensors, where some indices of  one array are paired with corresponding indices in a second array. This is typically  handled by first permuting (a.k.a. transposing) and reshaping the two input arrays such  that the contraction becomes equivalent to a matrix multiplication, which is then  performed by the highly efficient gemm method from BLAS. The resulting array might  need another reshape and index permutation to bring it in its final form.  Alternatively, a native Julia implementation that does not require the additional  transpositions (yet is typically slower) can be selected by using disable_blas()."
},

{
    "location": "#To-do-list-1",
    "page": "TensorOperations.jl",
    "title": "To do list",
    "category": "section",
    "text": "Make cache threadsafe.\nMake it easier to check contraction order and to splice in runtime information, or   optimize based on memory footprint or other custom cost functions."
},

{
    "location": "indexnotation/#",
    "page": "Index notation with @tensor macro",
    "title": "Index notation with @tensor macro",
    "category": "page",
    "text": ""
},

{
    "location": "indexnotation/#Index-notation-with-@tensor-macro-1",
    "page": "Index notation with @tensor macro",
    "title": "Index notation with @tensor macro",
    "category": "section",
    "text": "The prefered way to specify (a sequence of) tensor operations is by using the @tensor macro, which accepts an index notation format, a.k.a. Einstein notation (and in particular, Einstein\'s summation convention).This can most easily be explained using a simple example:using TensorOperations\nα=randn()\nA=randn(5,5,5,5,5,5)\nB=randn(5,5,5)\nC=randn(5,5,5)\nD=zeros(5,5,5)\n@tensor begin\n    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\n    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\nendIn the second to last line, the result of the operation will be stored in the preallocated array D, whereas the last line uses a different assignment operator := in order to define a new array E of the correct size. The contents of D and E will be equal.Following Einstein\'s summation convention, the result is computed by first tracing/ contracting the 3rd and 5th index of array A. The resulting array will then be contracted with array B by contracting its 2nd index with the last index of B and its last index with the first index of B. The resulting array has three remaining indices, which correspond to the indices a and c of array A and index b of array B (in that order). To this, the array C (scaled with α) is added, where its first two indices will be permuted to fit with the order a,c,b. The result will then be stored in array D, which requires a second permutation to bring the indices in the requested order a,b,c.In this example, the labels were specified by arbitrary letters or even longer names. Any valid variable name is valid as a label. Note though that these labels are never interpreted as existing Julia variables, but rather are converted into symbols by the @tensor macro. This means, in particular, that the specific tensor operations defined by the code inside the @tensor environment are completely specified at compile time. Alternatively, one can also choose to specify the labels using literal integer constants, such that also the following code specifies the same operation as above. Finally, it is also allowed to use primes (i.e. Julia\'s adjoint operator) to denote different indices, including using multiple subsequent primes.@tensor D[å\'\',ß,c\'] = A[å\'\',1,-3,c\',-3,2]*B[2,ß,1] + α*C[c\',å\'\',ß]The index pattern is analyzed at compile time and expanded to a set of calls to the basic tensor operations, i.e. add!, trace! and contract!. Temporaries are created where necessary, but will by default be saved to a global cache, so that they can be reused upon a next iteration or next call to the function in which the @tensor call is used. When experimenting in the REPL where every tensor expression is only used a single time, it might be better to use disable_cache(), though no real harm comes from using the cache (except higher memory usage). By default, the cache is allowed to take up to 50% of the total machine memory, though this is fully configurable.A contraction of several tensors A[a,b,c,d,e]*B[b,e,f,g]*C[c,f,i,j]*... is evaluted using pairwise contractions, using Julia\'s default left to right order, i.e. as ( (A[a,b,c,d,e] * B[b,e,f,g]) * C[c,f,i,j]) * .... However, if one respects the so-called NCON style of specifying indices, i.e. positive integers for the contracted indices and negative indices for the open indices, the different factors will be reordered and so that the pairwise tensor contractions contract over indices with smaller integer label first. For example,D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]will be evaluated as (A[-1,3,1,-2,2]*C[1,4,-4,-3])*B[3,2,4,-5]. Furthermore, in that case the indices of the output tensor (D in this case) do not need to be specified (using [:] instead), and will be chosen as (-1,-2,-3,-4,-5). Any other order is of course still possible by just specifying it.Furthermore, there is a @tensoropt macro which will optimize the contraction order to minimize the total number of multiplications (cost model might change or become configurable in the future). The optimal contraction order will be determined at compile time and will be hard coded in the macro expansion. The cost/size of the different indices can be specified in various ways, and can be integers or some arbitrary polynomial of an abstract variable, e.g. χ. In the latter case, the optimization assumes the assymptotic limit of large χ.@tensoropt D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for all indices (a,b,c,d,e,f)\n@tensoropt (a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for indices a,b,c,e, other indices (d,f) have cost 1\n@tensoropt !(a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost 1 for indices a,b,c,e, other indices (d,f) have cost χ\n@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)The optimal contraction tree as well as the associated cost can be obtained by@optimalcontractiontree C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]where the cost of the indices can be specified in the same various ways as for @tensoropt."
},

{
    "location": "functions/#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions/#TensorOperations.tensorcopy!",
    "page": "Functions",
    "title": "TensorOperations.tensorcopy!",
    "category": "function",
    "text": "tensorcopy!(A, IA, C, IC)\n\nCopies A into C by permuting the dimensions according to the pattern specified by IA and IC. Both iterables should contain the same elements in a different order. The result of this method is equivalent to permutedims!(C, A, p) where p is the permutation such that IC=IA[p]. The implementation of tensorcopy! is however more efficient on average, especially if Threads.nthreads() > 1.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensoradd!",
    "page": "Functions",
    "title": "TensorOperations.tensoradd!",
    "category": "function",
    "text": "tensoradd!(α, A, IA, β, C, IC)\n\nUpdates C to β*C + α * tensorcopy(A,IA,IC), but without creating the temporary permuted array.\n\nSee also: tensorcopy\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensortrace!",
    "page": "Functions",
    "title": "TensorOperations.tensortrace!",
    "category": "function",
    "text": "tensortrace!(α, A, IA, β, C, IC)\n\nUpdates C to β*C + α tensortrace(A,IA,IC), but without creating the temporary traced array.\n\nSee also: tensortrace\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcontract!",
    "page": "Functions",
    "title": "TensorOperations.tensorcontract!",
    "category": "function",
    "text": "tensorcontract!(α, A, labelsA, conjA, B, labelsB, conjB, β, C, labelsC)\n\nReplaces C with β C + α A * B, where some indices of array A are contracted with corresponding indices in array B by assigning them identical labels in the iterables labelsA and labelsB. The arguments conjA and conjB should be of type Char and indicate whether the data of arrays A and B, respectively, need to be conjugated (value \'C\') or not (value \'N\'). Every label should appear exactly twice in the union of labelsA, labelsB and labelsC, either in the intersection of labelsA and labelsB (for indices that need to be contracted) or in the interaction of either labelsA or labelsB with labelsC, for indicating the order in which the open indices should be match to the indices of the output array C.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorproduct!",
    "page": "Functions",
    "title": "TensorOperations.tensorproduct!",
    "category": "function",
    "text": "tensorproduct!(α, A, labelsA, B, labelsB, β, C, labelsC)\n\nReplaces C with β C + α A * B without any indices being contracted.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcopy",
    "page": "Functions",
    "title": "TensorOperations.tensorcopy",
    "category": "function",
    "text": "tensorcopy(A, IA, IC = IA)\n\nCreates a copy of A, where the dimensions of A are assigned indices from the iterable IA and the indices of the copy are contained in IC. Both iterables should contain the same elements in a different order.\n\nThe result of this method is equivalent to permutedims(A, p) where p is the permutation such that IC = IA[p]. The implementation of tensorcopy is however more efficient on average, especially if Threads.nthreads() > 1.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensoradd",
    "page": "Functions",
    "title": "TensorOperations.tensoradd",
    "category": "function",
    "text": "tensoradd(A, IA, B, IB, IC = IA)\n\nReturns the result of adding arrays A and B where the iterabels IA and IB denote how the array data should be permuted in order to be added. More specifically, the result of this method is equivalent to\n\ntensorcopy(A, IA, IC) + tensorcopy(B, IB, IC)\n\nbut without creating the temporary permuted arrays.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensortrace",
    "page": "Functions",
    "title": "TensorOperations.tensortrace",
    "category": "function",
    "text": "tensortrace(A, IA [, IC])\n\nTrace or contract pairs of indices of array A, by assigning them an identical indices in the iterable IA. The untraced indices, which are assigned a unique index, can be reordered according to the optional argument IC. The default value corresponds to the order in which they appear. Note that only pairs of indices can be contracted, so that every index in IA can appear only once (for an untraced index) or twice (for an index in a contracted pair).\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcontract",
    "page": "Functions",
    "title": "TensorOperations.tensorcontract",
    "category": "function",
    "text": "tensorcontract(A, IA, B, IB[, IC])\n\nContract indices of array A with corresponding indices in array B by assigning them identical labels in the iterables IA and IB. The indices of the resulting array correspond to the indices that only appear in either IA or IB and can be ordered by specifying the optional argument IC. The default is to have all open indices of array A followed by all open indices of array B. Note that inner contractions of an array should be handled first with tensortrace, so that every label can appear only once in IA or IB seperately, and once (for open index) or twice (for contracted index) in the union of IA and IB.\n\nThe contraction can be performed by a native Julia algorithm without creating any temporaries, or by first permuting the arrays such that the contraction becomes equivalent to a matrix product, which is then performed by BLAS. The latter is typically faster for large arrays. The choice of method is globally controlled by the methods enable_blas()](@ref) and disable_blas()](@ref).\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorproduct",
    "page": "Functions",
    "title": "TensorOperations.tensorproduct",
    "category": "function",
    "text": "tensorproduct(A, IA, B, IB, IC = (IA..., IB...))\n\nComputes the tensor product of two arrays A and B, i.e. returns a new array C with ndims(C) = ndims(A)+ndims(B). The indices of the output tensor are related to those of the input tensors by the pattern specified by the indices. Essentially, this is a special case of tensorcontract with no indices being contracted over. This method checks whether the indices indeed specify a tensor product instead of a genuine contraction.\n\n\n\n\n\n"
},

{
    "location": "functions/#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "The elementary tensor operations can also be accessed via functions, mainly for compatibility with older versions of this toolbox. The function-based syntax is also required when the contraction pattern is not known at compile time but is rather determined dynamically.These functions come in a mutating and non-mutating version. The mutating versions mimick the argument order of some of the BLAS functions, such as blascopy!, axpy! and gemm!. Symbols A and B always refer to input arrays, whereas C is used to denote the array where the result will be stored. They also return C and are therefore type stable. The greek letters α and β denote scalar coefficients.tensorcopy!\ntensoradd!\ntensortrace!\ntensorcontract!\ntensorproduct!The non-mutating functions are simpler in not allowing scalar coefficients and conjugation. They also take a default value for the labels of the output array if these are not specified. However, the return type is only inferred if the labels are entered as tuples, and also IC is specified. They are simply called as:tensorcopy\ntensoradd\ntensortrace\ntensorcontract\ntensorproduct"
},

{
    "location": "cache/#",
    "page": "Cache for temporaries",
    "title": "Cache for temporaries",
    "category": "page",
    "text": ""
},

{
    "location": "cache/#Cache-for-temporaries-1",
    "page": "Cache for temporaries",
    "title": "Cache for temporaries",
    "category": "section",
    "text": "Contracting a sequence of tensors is provably most efficient (in terms of number of computations) by contracting them pairwise. However, this requires that several intermediate results need to be stored. In addition, if the contraction needs to be performed as a BLAS matrix multiplication (which is typically the fastest choice), every tensor typically needs an additional permuted copy that is compatible with the implementation of the contraction as multiplication. All these temporary arrays, which can be large, put a a lot of pressure on Julia\'s garbage collector, and the total time spent in the garbage collector can become significant.That\'s why there is now a functionality to store intermediate results in a package wide cache, where they can be reused upon a next run, either a next iteration if the tensor contraction appears within the body of a loop, or on the next function call if it appears directly within a given function. This mechanism only works with the @tensor macro, not with the function-based interface.The @tensor macro expands the given expression and immediately generates the code to create the necessary temporaries. It associates with each of them a random symbol (gensym()) and uses this as an identifier in a package wide global cache structure TensorOperations.cache, the implementation of which is a least-recently used cache dictionary borrowed from the package LRUCache.jl, but adapted in such a way that it uses a maximum memory size rather than a maximal number of objects. Thereto, it estimates the size of each object added to the cache using Base.summarysize and, when discards objects once a certain memory limit is reached."
},

{
    "location": "cache/#TensorOperations.enable_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.enable_cache",
    "category": "function",
    "text": "enable_cache(; maxsize::Int = ..., maxrelsize::Real = 0.5)\n\n(Re)-enable the cache for further use; set the maximal size maxsize (as number of bytes) or relative size maxrelsize, as a fraction between 0 and 1, resulting in maxsize = floor(Int, maxrelsize * Sys.total_memory()).\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.disable_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.disable_cache",
    "category": "function",
    "text": "disable_cache()\n\nDisable the cache for further use but does not clear its current contents. Also see clear_cache()\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.cachesize",
    "page": "Cache for temporaries",
    "title": "TensorOperations.cachesize",
    "category": "function",
    "text": "cachesize()\n\nReturn the current memory size (in bytes) of all the objects in the cache.\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.clear_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.clear_cache",
    "category": "function",
    "text": "clear_cache()\n\nClear the current contents of the cache.\n\n\n\n\n\n"
},

{
    "location": "cache/#Enabling-and-disabling-the-cache-1",
    "page": "Cache for temporaries",
    "title": "Enabling and disabling the cache",
    "category": "section",
    "text": "The use of the cache can be enabled or disabled usingenable_cache\ndisable_cacheFurthermore, the current total size of all the objects stored in the cache can be obtained using the method cachesize, and clear_cache can be triggered to release all the objects currently stored in the cache, such that they can be removed by Julia\'s garbage collector.cachesize\nclear_cache"
},

{
    "location": "cache/#Cache-and-multithreading-1",
    "page": "Cache for temporaries",
    "title": "Cache and multithreading",
    "category": "section",
    "text": "The LRU cache currently used is not thread safe, i.e. if there is any chance that different threads will run tensor expressions using the @tensor environment, you should disable_cache()."
},

{
    "location": "implementation/#",
    "page": "Implementation",
    "title": "Implementation",
    "category": "page",
    "text": ""
},

{
    "location": "implementation/#Implementation-1",
    "page": "Implementation",
    "title": "Implementation",
    "category": "section",
    "text": ""
},

{
    "location": "implementation/#Index-notation-and-the-@tensor-macro-1",
    "page": "Implementation",
    "title": "Index notation and the @tensor macro",
    "category": "section",
    "text": "We start by describing the implementation of the @tensor and @tensoropt macro. The macros end up transforming the index expression to the corresponding function calls to the primitive building blocks, which are discussed in the next section. In principle, anyone interested in making the @tensor macro work for custom array types should only reimplement these building blocks, but it is useful to understand how the tensor expressions are processed."
},

{
    "location": "implementation/#Tensors,-a.k.a-indexed-objects-1",
    "page": "Implementation",
    "title": "Tensors, a.k.a indexed objects",
    "category": "section",
    "text": "The central objects in tensor expressions that follow the @tensor macro are the index expressions of the form A[a,b,c].  These are detected as subexpressions of the form Expr(:ref, args). In fact, what is recognized by @tensor as an indexed object is more general, and also includes expressions of the form A[a b c] or A[a b; c d e] (Expr(:typed_hcat, args) and Expr(:typed_vcat, args)). The last form in particular is useful in more general contexts, and allows to distinguish between two sets of indices, referred to as left (a and b) and right (c, d and e) indices. This can be used when the @tensor macro wants to be generalized to user types, which for example distinguish between contravariant (upper) and covariant (lower) indices. For AbstractArray subtypes, such distinction is of course meaningless.Note that the object being indexed, i.e. A in all of the above examples or args[1] in the corresponding Expr objects, can itself be an expression, that is not further analyzed. In particular, this may itself contain further indexing objects, such that one can get tensor objects from a list, e.g. list[3][a,b,c] or slice an array before using it in the @tensor expression, e.g. A[1:2:end,:,3:end][a,b,c].Everything appearing in the [ ], e.g. args[2:end] (in case of :ref or :typed_hcat, the argument structure of :typed_vcat is slightly more complicated) is considered to be a valid index. This can be any valid Julia variable name, which is just kept as symbol, or any literal integer constant, (for legacy reasons, any literal character constant,) or finally an Expr(Symbol(\"\'\"),args), i.e. an expression of the form :(a\'). The latter is converted to a symbol of the form Symbol(\"a′\") when a is itself a symbol or integer, or this is applied recursively if a contains more primes.The implementation for detecting tensors and indices (istensor, isindex) and actually converting them to a useful format (maketensor, makeindex) are found in src/indexnotation/tensorexpressions.jl. In particular, maketensor will return the indexed object, which is just esc(args[1]), the list of left indices and the list of right indices.Furthermore, there is isscalar and makescalar to detect and process subexpressions that will evaluate to a scalar. Finally, there is isgeneraltensor and makegeneraltensor to detect and process a a tensor (indexed object), that is possibly conjugated or multiplied with a scalar. This is useful because the primitive tensor operations (i.e. see Building blocks below), accept a scalar factor and conjugation flag, so that these operations can be done simultaneously and do not need to be evaluated at first (which would require additional temporaries). The function makegeneraltensor in particular will return the indexed object, the list of left indices, the list of right indices, the scalar factor, and a flag (Bool) that indicates whether the object needs to be conjugated (true) or not (false).The file src/indexnotation/tensorexpressions.jl also contains simple methods to detect assignment (isassignment) into existing objects (i.e. =, += and -=) or so-called definitions (isdefinition), that create a new object (via := or its Unicode variant ≔, obtained as \\coloneq + TAB). The function getlhsrhs will return the left hand side and right hand side of an assignment or definition expression separately.Finally, there are methods to detect whether the right hand side is a valid tensor expression (istensorexpr) and to get the indices of a complete tensor expressions. In particular,  getindices returns a list of indices that will remain after the expression is evaluated (i.e. any index that is not contracted in the expression because it only appears once), whereas getallindices returns a list of all indices that appear in the expression. The latter is used to analyze complete tensor contraction graphs."
},

{
    "location": "implementation/#The-macros-@tensor-and-@tensoropt-1",
    "page": "Implementation",
    "title": "The macros @tensor and @tensoropt",
    "category": "section",
    "text": "Actual processing of the complete expression that follows the @tensor macro and converting it into a list of actual calls to the primitive tensor operations is handled by the functions defined in src/indexnotation/tensormacro.jl. The integral expression received by the @tensor macro is passed on to the tensorify function. The @tensoropt macro will first generate the data required to optimize contraction order, by calling optdata. If no actual costs are specified, i.e. @tensoropt receives a single expression, then optdata just assigns the same cost to all indices in the expression. Otherwise, the expression that specifies the costs need to be parsed first (parsecost). Finally, @tensoropt also calls tensorify passing the optdata as a second optional argument (whose default value nothing is used by @tensor).The function tensorify consists of several steps. Firstly, it canonicalizes the expression. Currently, this involves a single pass which expands all conj calls of e.g. products or sums to a conj call on each of the arguments (via the expandconj function). Secondly, tensorify will process the contraction order using processcontractorder. This starts from the fact that a product of several tensors, specified as A[...]*B[...]*C[...] yields a single Expr(:call,[:*,...]) object where all the factors are still together. If a user wants a specific order, he can do so by grouping them with parenthesis. Whenever an expression Expr(:call,[:*,...]) is found where more than two of the arguments satisfy isgeneraltensor, processcontractorder will convert it into a nested set of pairwise multiplications according to a number of strategies discussed in the next subsection.The major part of tensorify is to generate the correct function calls corresponding to the tensor expression. It detects assignments or definitions (the most common case) and validates the left hand side thereof (i.e. it should satisfy istensor and have no duplicate indices). Then, it generates the corresponding function calls corresponding to the index expression by passing onto the deindexify function, which takes the signature julia deindexify(dst, β, ex, α, leftind, rightind, istemporary = false) Here, dst is the symbol or expression corresponding to the destination object; it\'s nothing in case of a definition (:=), i.e. if the object corresponding to the result needs to be created/allocated. β and α are isscalar expressions, and deindexify will create the function calls required to update dst with multiplying dst with β and adding α times the result of the expression ex to it; this is supported as a one step process by each of the primitive operations. leftind and rightind correspond to the list of indices of the left hand side of the defition or assignment. The final argument istemporary indicates, if dst == nothing and a new object needs to be created/allocated, whether it is a temporary object. If istemporary == true, it can be stored in the cache and later retrieved. If istemporary == false, it corresponds to an explicit left hand side created by the user in a definition, and should not be in the cache.The function deindexify will determine the top level operation represented by ex (which should be a istensorexpr), and then pass on to  deindexify_generaltensor, deindexify_linearcombination, deindexify_contraction for actually creating the correct function call expressions. If any of the arguments of e.g. a linear combination or a tensor contraction is itself a composite tensor expression (i.e. not a isgeneraltensor), deindexify is called recursively."
},

{
    "location": "implementation/#Analyzing-contraction-graphs-(a.k.a-tensor-networks)-and-optimizing-contraction-order-1",
    "page": "Implementation",
    "title": "Analyzing contraction graphs (a.k.a tensor networks) and optimizing contraction order",
    "category": "section",
    "text": "The function processcontractorder, which is excuted before the index expression is converted to function calls, will detect any multiplication with more than two isgeneraltensor factors, and divide it up into a nested sequence of pairwise multiplications (tensor contractions), i.e. a tree. If the @tensor macro was used, optdata = nothing and in principle the multiplication will be performed from left to right. There is one exception, which is that if the indices follow the NCON convention, i.e. negative integers are used for uncontracted indices and positive integers for contracted indices. Then the contraction tree is built such that tensors that share the contraction index which is the lowest positive integer are contracted first. Relevant code can be found in src/indexnotation/ncontree.jlWhen the @tensoropt macro was used, optdata is a dictionary associating a cost (either a number or a polynomial in some abstract scaling parameter) to every index, and this information is used to determine the (asymptotically) optimal contraction tree (in terms of number of floating point operations). The code for the latter is in src/indexnotation/optimaltree.jl, with the lightweight polynomial implementation in src/indexnotation/polynomial.jl. Aside from a generic polynomial type Poly, the latter also contains a Power type which represents a single term of a polynomial (i.e. a scalar coefficient and an exponent). This type is closed under multiplication, and can be multiplied much more efficiently. Only under addition is a generic Poly returned."
},

{
    "location": "implementation/#TensorOperations.add!",
    "page": "Implementation",
    "title": "TensorOperations.add!",
    "category": "function",
    "text": "add!(α, A, conjA, β, C, indleft, indright)\n\nImplements C = β*C+α*permute(op(A)) where A is permuted such that the left (right) indices of C correspond to the indices indleft (indright) of A, and op is conj if conjA == :C or the identity map if conjA == :N (default). Together, (indleft..., indright...) is a permutation of 1 to the number of indices (dimensions) of A.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.trace!",
    "page": "Implementation",
    "title": "TensorOperations.trace!",
    "category": "function",
    "text": "trace!(α, A, conjA, β, C, indleft, indright, cind1, cind2)\n\nImplements C = β*C+α*partialtrace(op(A)) where A is permuted and partially traced, such that the left (right) indices of C correspond to the indices indleft (indright) of A, and indices cindA1 are contracted with indices cindA2. Furthermore, op is conj if conjA == :C or the identity map if conjA=:N (default). Together, (indleft..., indright..., cind1, cind2) is a permutation of 1 to the number of indices (dimensions) of A.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.contract!",
    "page": "Implementation",
    "title": "TensorOperations.contract!",
    "category": "function",
    "text": "contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indleft, indright, syms = nothing)\n\nImplements C = β*C+α*contract(opA(A),opB(B)) where A and B are contracted, such that the indices cindA of A are contracted with indices cindB of B. The open indices oindA of A and oindB of B are permuted such that C has left (right) indices corresponding to indices indleft (indright) out of (oindA..., oindB...). The operation opA (opB) acts as conj if conjA (conjB) equal :C or as the identity map if conjA (conjB) equal :N. Together, (oindA..., cindA...) is a permutation of 1 to the number of indices of A and (oindB..., cindB...) is a permutation of 1 to the number of indices of C. Furthermore, length(cindA) == length(cindB), length(oindA)+length(oindB) equals the number of indices of C and (indleft..., indright...) is a permutation of 1 ot the number of indices of C.\n\nThe final argument syms is optional and can be either nothing, or a tuple of three symbols, which are used to identify temporary objects in the cache to be used for permuting A, B and C so as to perform the contraction as a matrix multiplication.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.checked_similar_from_indices",
    "page": "Implementation",
    "title": "TensorOperations.checked_similar_from_indices",
    "category": "function",
    "text": "checked_similar_from_indices(C, T, indleft, indright, A, conjA = :N)\n\nReturns an object similar to A which has an eltype given by T and whose left indices correspond to the indices indleft from op(A), and its right indices correspond to the indices indright from op(A), where op is conj if conjA == :C or does nothing if conjA == :N (default). Here, C is a potential candidate for the similar object. If C === nothing, or its its eltype or shape does not match, a new object is allocated and returned. Otherwise, C is returned.\n\n\n\n\n\nchecked_similar_from_indices(C, T, indoA, indoB, indleft, indright,\n                                A, B, conjA = :N, conjB= :N)\n\nReturns an object similar to A which has an eltype given by T and dimensions/sizes corresponding to a selection of those of opA(A) and opB(B) concatenated. Out of the collection of indices in indoA of opA(A) and indoB of opB(B), we construct an object whose left (right) indices correspond to indices indleft (indright) from that collection. Furthermore, C is a potential candidate for the similar object. If C === nothing, or its its eltype or shape does not match, a new object is allocated and returned. Otherwise, C is returned.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.scalar",
    "page": "Implementation",
    "title": "TensorOperations.scalar",
    "category": "function",
    "text": "scalar(C)\n\nReturns the single element of a tensor-like object with zero indices or dimensions.\n\n\n\n\n\n"
},

{
    "location": "implementation/#Building-blocks-1",
    "page": "Implementation",
    "title": "Building blocks",
    "category": "section",
    "text": "The @tensor macro converts the index expression into a set of function calls corresponding to three primitive operations: addition, tracing and contraction. These operations are implemented for arbitrary strided arrays from Julia Base, i.e. Arrays, views with ranges thereof, and certain reshape operations. This includes certain arrays that can only be determined to be strided on runtime, and does therefore not coincide with the type union StridedArray from Julia Base. In fact, the methods accept AbstractArray objects, but convert these to (Unsafe)StridedView objects from the package Strided.jl, and we refer to this package for a more detailed discussion on which arrays are supported and why.The primitive tensor operations are captured by the following mutating methods (note that these are not exported)TensorOperations.add!\nTensorOperations.trace!\nTensorOperations.contract!These are the central objects that should be overloaded by custom tensor types that would like to be used within the @tensor environment. They are also used by the function based methods discussed in the section Functions.Furthermore, it is essential to be able to construct new tensor objects that are similar to existing ones, i.e. to place the result of the computation in case no output is specified. In order to reuse temporary objects stored in the global cache, this method also receives a candidate similar object, which it can return if it matches the requirements.TensorOperations.checked_similar_from_indicesNote that the type of the cached object is not known to the compiler, as the cache stores objects as Any. Therefore, the function checked_similar_from_indices should try to restore the type information. By passing any object retrieved from the cache through this function, type stability within the @tensor macro can then still be guaranteed.Finally, there is a particularly simple method scalar whose sole purpose is to extract the single entry of an object with zero indices, i.e. an instance of AbstractArray{T,0} in case of Julia Base arrays:TensorOperations.scalarThe implementation of all of these methods can be found in src/implementation/stridedarray.jl.By implementing these five methods for other types that represent some kind of tensor or multidimensional object, they can be used in combination with the @tensor macro. In particular, we also provide basic support for contracting a Diagonal matrix with an arbitrary strided array in src/implementation/diagonal.jl."
},

]}
