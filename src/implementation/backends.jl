const _backendType = Dict{Symbol,Type{<:AbstractBackend}}()
const _backendSymbol = Dict{DataType,Symbol}()
const _backends = Symbol[]
const _backend_packages = Dict{Symbol,Tuple{Vararg{Symbol}}}()
const _initialized_backends = Set{Symbol}()

function _backend_instance(sym::Symbol)::AbstractBackend
    if haskey(_backendType, sym)
        return _backendType[sym]()
    end
    throw(ArgumentError("Unsupported backend $sym"))
end

"Returns a list of supported backends"
backends() = _backends

"Returns the name of the current backend"
backend_name() = CURRENT_BACKEND.sym

#===========================================================================================
Supported Backends
===========================================================================================#

macro init_backend(name, deps...)
    package_str = string.(deps)
    str = string(name)
    T = Symbol(str * "Backend")
    return esc(quote
                   Symbol($str) ∈ TensorOperations._backends && @warn "redefinition of backend `$($str)`"
                   struct $T <: TensorOperations.AbstractBackend end
                   export $T
                   TensorOperations.backend_name(::$T) = Symbol($str)
                   push!(TensorOperations._backends, Symbol($str))
                   TensorOperations._backendType[Symbol($str)] = $T
                   TensorOperations._backendSymbol[$T] = Symbol($str)
                   TensorOperations._backend_packages[Symbol($str)] = Symbol.($package_str)
               end)
end

@init_backend None
@init_backend Strided Strided
@init_backend StridedBLAS Strided
@init_backend CUDA CUDA cuTENSOR

@init_backend Julia
@init_backend Cache

const _project = Pkg.Types.read_package(normpath(@__DIR__, "../..", "Project.toml"))
const _compats = _project.compat

function _check_installed(backend::Union{Module,Symbol,AbstractString}; warn=true)
    sym = Symbol((string(backend)))
    if warn && !haskey(_backend_packages, sym)
        @warn "backend `$sym` is not compatible with TensorOperations"
        return [nothing]
    end

    versions = map(get(_backend_packages, sym, (backend,))) do dep_pkg
        str = string(dep_pkg)
        # check installed
        dep_pkg_id = Base.identify_package(str)
        version = if isnothing(dep_pkg_id)
            nothing
        else
            get(Pkg.dependencies(), dep_pkg_id.uuid, (; version=nothing)).version
        end
        if isnothing(version)
            warn && @warn "backend `$sym` dependency `$str` is not installed."
        else
            # check compatibility
            if haskey(_compats, str)
                if (be_c = _compats[str]) isa String  # julia 1.6
                    if version ∉ Pkg.Types.semver_spec(be_c)
                        @warn "`$str` $version is not compatible with this version of `TensorOperations`. The declared compatibility is $(be_c)."
                    end
                else
                    if isempty(intersect(version, be_c.val))
                        @warn "`$str` $version is not compatible with this version of `TensorOperations`. The declared compatibility is $(be_c.str)."
                    end
                end
            end
        end
        return version
    end
    return versions
end

function _check_compat(m::Union{Module,Symbol,AbstractString}; warn=true)
    (be_vs = _check_installed(m; warn))
    for be_v in be_vs
        isnothing(be_v) && continue
        if (be_c = _compats[string(m)]) isa String  # julia 1.6
            if be_v ∉ Pkg.Types.semver_spec(be_c)
                @warn "`$m` $be_v is not compatible with this version of `TensorOperations`. The declared compatibility is $(be_c)."
            end
        else
            if isempty(intersect(be_v, be_c.val))
                @warn "`$m` $be_v is not compatible with this version of `TensorOperations`. The declared compatibility is $(be_c.str)."
            end
        end
    end
    return nothing
end

initialized(sym::Symbol) = sym ∈ _initialized_backends

function _initialize_backend(pkg::AbstractBackend)
    sym = backend_name(pkg)
    for pkg_dep in get(_backend_packages, sym, nothing)
        @info "Initializing $pkg_dep"
        @eval Main begin
            using $pkg_dep: $pkg_dep
        end
    end
    _check_installed(sym)
    push!(_initialized_backends, sym)
    return nothing
end

#===========================================================================================
Current Backend
===========================================================================================#

mutable struct CurrentBackend
    sym::Symbol
    backend::AbstractBackend
end
CurrentBackend(sym::Symbol) = CurrentBackend(sym, _backend_instance(sym))

const CURRENT_BACKEND = CurrentBackend(:None)
const CURRENT_ALLOCATOR = CurrentBackend(:None)

function backend()
    CURRENT_BACKEND.sym === :none && load_default_backend()
    return CURRENT_BACKEND.backend
end

function backend(sym::Symbol)
    if sym in _backends
        return backend(_backend_instance(sym))
    else
        @warn "`:$sym` is not a supported backend."
        return CURRENT_BACKEND.backend
    end
end

function backend(backend::AbstractBackend)
    sym = backend_name(backend)
    initialized(sym) || _initialize_backend(backend)

    CURRENT_BACKEND.sym = sym
    CURRENT_BACKEND.backend = backend
    @info "default backend set to `$sym`"
    return backend
end

function allocator()
    CURRENT_ALLOCATOR.sym === :none && load_default_allocator()
    return CURRENT_ALLOCATOR.backend
end

function allocator(sym::Symbol)
    if sym in _backends
        return allocator(_backend_instance(sym))
    else
        @warn "`:$sym` is not a supported backend."
        return CURRENT_ALLOCATOR.backend
    end
end

function allocator(backend::AbstractBackend)
    sym = backend_name(backend)
    initialized(sym) || _initialize_backend(backend)

    CURRENT_ALLOCATOR.sym = sym
    CURRENT_ALLOCATOR.backend = backend
    @info "default allocator set to `$sym`"
    return backend
end

#===========================================================================================
Default Backend
===========================================================================================#

const TENSOROPERATIONS_DEFAULT_BACKEND = load_preference(TensorOperations,
                                                         "default_backend", "None")
const TENSOROPERATIONS_DEFAULT_ALLOCATOR = load_preference(TensorOperations,
                                                           "default_allocator", "Julia")

function load_default_backend()
    sym = Symbol(get(ENV, "TENSOROPERATIONS_DEFAULT_BACKEND",
                     TENSOROPERATIONS_DEFAULT_BACKEND))
    return backend(sym)
end

function load_default_allocator()
    sym = Symbol(get(ENV, "TENSOROPERATIONS_DEFAULT_ALLOCATOR",
                     TENSOROPERATIONS_DEFAULT_ALLOCATOR))
    return allocator(sym)
end

function set_default_backend!(backend::Union{Nothing,AbstractString,Symbol}=nothing;
                              kwargs...)
    if isnothing(backend)
        delete_preferences!(TensorOperations, "default_backend", kwargs...)
    else
        if _check_installed((value = string(backend))) !== nothing
            set_preferences!(TensorOperations, "default_backend" => value; kwargs...)
        end
    end
    return nothing
end

function set_default_allocator!(backend::Union{Nothing,AbstractString,Symbol}=nothing;
                                kwargs...)
    if isnothing(backend)
        delete_preferences!(TensorOperations, "default_allocator", kwargs...)
    else
        if _check_installed((value = string(backend))) !== nothing
            set_preferences!(TensorOperations, "default_allocator" => value; kwargs...)
        end
    end
    return nothing
end

#===========================================================================================
Backend options
===========================================================================================#

# A switch for enabling/disabling the use of BLAS for tensor contractions
const _use_blas = Ref{Bool}(load_preference(TensorOperations, "use_blas", false))
use_blas() = _use_blas[]
function disable_blas()
    _use_blas[] = false
    return
end
function enable_blas()
    _use_blas[] = true
    return
end

#===========================================================================================
Backend insertion
===========================================================================================#

function tensoradd!(C, A, pA::Index2Tuple, conjA, α, β)
    return tensoradd!(backend(), C, A, pA, conjA, α, β)
end

function tensorcontract!(C, pC::Index2Tuple, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple,
                         conjB, α, β)
    return tensorcontract!(backend(), C, pC, A, pA, conjA, B, pB, conjB, α, β)
end

function tensortrace!(C, pC::Index2Tuple, A, pA::Index2Tuple, conjA, α, β)
    return tensortrace!(backend(), C, pC, A, pA, conjA, α, β)
end
