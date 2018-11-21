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
    "text": "A macro @tensor for conveniently specifying tensor contractions and index permutations   via Einstein\'s index notation convention. The index notation is analyzed at compile time.\nAbility to optimize pairwise contraction order   using the @tensoropt macro.\nSupport for any Julia Base array which qualifies as strided, i.e. such that its entries   are layed out according to a regular pattern in memory. The only exception are ReinterpretedArray   objects (implementation provided by Strided.jl, see below).\nImplementation can easily be extended to other types, by overloading a small set of methods.\nEfficient implementation of a number of basic tensor operations (see below), by relying   on Strided.jl and gemm from BLAS for contractions.   The latter is optional but on by default, it can be controlled by a package wide setting   via enable_blas() and disable_blas(). If BLAS is disabled or cannot be applied (e.g.   non-matching or non-standard numerical types), Strided.jl is also used for the contraction.\nA package wide cache for storing temporary arrays that are generated when evaluating complex   tensor expressions within the @tensor macro (based on the implementation of   LRUCache)."
},

{
    "location": "#Tensor-operations-1",
    "page": "TensorOperations.jl",
    "title": "Tensor operations",
    "category": "section",
    "text": "TensorOperations.jl is centered around 3 basic tensor operations, i.e. primitives in which every more complicated tensor expression is deconstructed.addition: Add a (possibly scaled version of) one array to another array, where the  indices of the both arrays might appear in different orders. This operation combines normal  array addition and index permutation. It includes as a special case copying one array into  another with permuted indices.\nThe actual implementation is provided by Strided.jl,  which contains multithreaded implementations and cache-friendly blocking strategies for  an optimal efficiency.\ntrace or inner contraction: Perform a trace/contraction over pairs of indices of an array,  where the result is a lower-dimensional array. As before, the actual implementation is  provided by Strided.jl.\ncontraction: Performs a general contraction of two tensors, where some indices of one  array are paired with corresponding indices in a second array. This is typically handled  by first permuting (a.k.a. transposing) and reshaping the two input arrays such that the  contraction becomes equivalent to a matrix multiplication, which is then performed by the  highly efficient gemm method from BLAS. The resulting array might need another reshape  and index permutation to bring it in its final form. Alternatively, a native Julia  implementation that does not require the additional transpositions (yet is typically slower)  can be selected by using disable_blas()."
},

{
    "location": "#To-do-list-1",
    "page": "TensorOperations.jl",
    "title": "To do list",
    "category": "section",
    "text": "Make cache threadsafe.\nMake it easier to check contraction order and to splice in runtime information, or optimize   based on memory footprint or other custom cost functions."
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
    "text": "The prefered way to specify (a sequence of) tensor operations is by using the @tensor macro, which accepts an index notation format, a.k.a. Einstein notation (and in particular, Einstein\'s summation convention).This can most easily be explained using a simple example:using TensorOperations\nα=randn()\nA=randn(5,5,5,5,5,5)\nB=randn(5,5,5)\nC=randn(5,5,5)\nD=zeros(5,5,5)\n@tensor begin\n    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\n    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\nendIn the second to last line, the result of the operation will be stored in the preallocated array D, whereas the last line uses a different assignment operator := in order to define a new array E of the correct size. The contents of D and E will be equal.Following Einstein\'s summation convention, the result is computed by first tracing/contracting the 3rd and 5th index of array A. The resulting array will then be contracted with array B by contracting its 2nd index with the last index of B and its last index with the first index of B. The resulting array has three remaining indices, which correspond to the indices a and c of array A and index b of array B (in that order). To this, the array C (scaled with α) is added, where its first two indices will be permuted to fit with the order a,c,b. The result will then be stored in array D, which requires a second permutation to bring the indices in the requested order a,b,c.In this example, the labels were specified by arbitrary letters or even longer names. Any valid variable name is valid as a label. Note though that these labels are never interpreted as existing Julia variables, but rather are converted into symbols by the @tensor macro. This means, in particular, that the specific tensor operations defined by the code inside the @tensor environment are completely specified at compile time. Alternatively, one can also choose to specify the labels using literal integer or character constants, such that also the following code specifies the same operation as above. Finally, it is also allowed to use primes (i.e. Julia\'s adjoint operator) to denote different indices.@tensor D[å,ß,c\'] = A[å,1,\'f\',c\',\'f\',2]*B[2,ß,1] + α*C[c\',å,ß]The index pattern is analyzed at compile time and expanded to a set of calls to the basic tensor operations, i.e. add!, trace! and contract!. Temporaries are created where necessary, but will by default be saved to a global cache, so that they can be reused upon a next iteration or next call to the function in which the @tensor call is used. When experimenting in the REPL, it might be better to use disable_cache().By default, a contraction of several tensors A[a,b,c,d,e]*B[b,e,f,g]*C[c,f,i,j]*... is evaluted using pairwise contractions from left to right, i.e. as ( (A[a,b,c,d,e] * B[b,e,f,g]) * C[c,f,i,j]) * .... However, if one respects the so-called NCON style of specifying indices, i.e. positive integers for the contracted indices and negative indices for the open indices, the different factors will be reordered and so that the pairwise tensor contractions contract over indices with smaller integer label first. For example,D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]will be evaluated as (A[-1,3,1,-2,2]*C[1,4,-4,-3])*B[3,2,4,-5]. Furthermore, in that case the indices of the output tensor (D in this case) do not need to be specified (using [:] instead), and will be chosen as (-1,-2,-3,-4,-5). Any other order is of course still possible by just specifying it.Furthermore, there is a @tensoropt macro which will optimize the contraction order to minimize the total number of multiplications (cost model might change or become choosable in the future). The optimal contraction order will be determined at compile time and will be hard coded in the macro expansion. The cost/size of the different indices can be specified in various ways, and can be integers or some arbitrary polynomial of an abstract variable, e.g. χ. In the latter case, the optimization assumes the assymptotic limit of large χ.@tensoropt D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for all indices (a,b,c,d,e,f)\n@tensoropt (a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for indices a,b,c,e, other indices (d,f) have cost 1\n@tensoropt !(a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost 1 for indices a,b,c,e, other indices (d,f) have cost χ\n@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)The optimal contraction tree as well as the associated cost can be obtained by@optimalcontractiontree C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]where the cost of the indices can be specified in the same various ways as for @tensoropt."
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
    "text": "Contracting a sequence of tensors is provably most efficient (in terms of number of computations) by contracting them pairwise. However, this requires that several intermediate results need to be stored. In addition, if the contraction needs to be performed as a BLAS matrix multiplication (which is typically the fastest choice), every tensor typically needs an additional permuted copy that is compatible with the implementation of the contraction as multiplication. All these temporary arrays, which can be large, put a a lot of pressure on Julia\'s garbage collector, and the total time spent in the garbage collector can become significant.That\'s why there is now a functionality to store intermediate results in a package wide cache, where they can be reused upon a next run, either a next iteration if the tensor contraction appears within the body of a loop, or on the next function call if it appears directly within a given function. This mechanism only works with the @tensor macro, not with the function-based interface.The @tensor macro expands the given expression and immediately generates the code to create the necessary temporaries. It associates with each of them a random symbol (gensym()) and uses this as an identifier in a package wide global cache structure CACHE, the implementation of which is a least-recently used cache dictionary borrowed from the package LRUCache.jl, but adapted in such a way that it uses a maximum memory size rather than a maximal number of objects. Thereto, it estimates the size of each object added to the cache using Base.summarysize and, when discards objects once a certain memory limit is reached."
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
    "text": "The use of the cache can be enabled or disabled usingenable_cache\ndisable_cache\nclear_cache"
},

{
    "location": "cache/#Cache-and-multithreading-1",
    "page": "Cache for temporaries",
    "title": "Cache and multithreading",
    "category": "section",
    "text": "The LRU cache currently used is not thread safe, i.e. if there is any chance that different threads will run the same @tensor expression block, you should disable_cache()."
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
    "location": "implementation/#Building-blocks-1",
    "page": "Implementation",
    "title": "Building blocks",
    "category": "section",
    "text": "Under the hood, the implementation is  centered around the primitive operations: addition, tracing and contraction. These operations are implemented for arbitrary strided arrays from Julia Base, i.e. Arrays, views with ranges thereof, and certain reshape operations. This includes certain arrays that can only be determined to be strided on runtime, and does therefore not coincide with the type union StridedArray from Julia Base. In fact, the methods accept AbstractArray objects, but convert these to (Unsafe)StridedView objects from the package Strided.jl, and we refer to this package for a more detailed discussion of which arrays are supported and why.Nonetheless, the implementation can easily be extended to user defined types, especially if they just wrap multidimensional data with a strided memory storage. The building blocks resemble the functions discussed above, but have a different interface and are more general. They are used both by the functions as well as by the @tensor macro, as discussed below. Note that these functions are not exported.add!(α, A, conjA, β, C, indCinA)\nImplements C = β*C+α*permute(op(A)) where A is permuted according to indCinA and op is conj if conjA=Val{:C} or the identity map if conjA=Val{:N}. The indexable collection indCinA contains as nth entry the dimension of A associated with the nth dimension of C.\ntrace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)\nImplements C = β*C+α*partialtrace(op(A)) where A is permuted and partially traced, according to indCinA, cindA1 and cindA2, and op is conj if conjA=Val{:C} or the identity map if conjA=Val{:N}. The indexable collection indCinA contains as nth entry the dimension of A associated with the nth dimension of C. The partial trace is performed by contracting dimension cindA1[i] of A with dimension cindA2[i] of A for all i in 1:length(cindA1).\ncontract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, [method])\nImplements C = β*C+α*contract(op(A),op(B)) where A and B are contracted according to oindA, cindA, oindB, cindB and indCinoAB. The operation op acts as conj if conjA or conjB equal Val{:C} or as the identity map if conjA (conjB) equal Val{:N}. The dimension cindA[i] of A is contracted with dimension cindB[i] of B. The nth dimension of C is associated with an uncontracted (open) dimension of A or B according to indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA] with NoA=length(oindA) the number of open dimensions of A.\nThe optional argument method specifies whether the contraction is performed using BLAS matrix multiplication by specifying Val{:BLAS}, or using a native algorithm by specifying Val{:native}. The native algorithm does not copy the data but is typically slower. The BLAS-based algorithm is chosen by default, if the element type of the output array is in Base.LinAlg.BlasFloat."
},

{
    "location": "implementation/#Index-notation-and-the-@tensor-macro-1",
    "page": "Implementation",
    "title": "Index notation and the @tensor macro",
    "category": "section",
    "text": ""
},

]}
