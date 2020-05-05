#### Reference state profiles

export VirtualTemperature

"""
    AbstractReferenceState{FT<:AbstractFloat}

Reference states which can then be called with the functor:

```
(ref_state::AbstractReferenceState)(param_set::AbstractParameterSet, z::FT) {where FT<:AbstractFloat}
```
where
 - `param_set` parameter set, used to dispatch planet parameter function calls
 - `z` vertical (altitude) coordinate
or
 - `e_pot` potential energy (e.g., gravitational potential)
"""
abstract type AbstractReferenceState{FT<:AbstractFloat} end

"""
    VirtualTemperature{FT}

A virtual temperature profile that decays smoothly with height `z`, dropping by a specified temperature difference `ΔTv` over a height scale `H_t`.

```math
Tv(z) = \\max(Tv{\\text{surface}} − ΔTv \\tanh(z/H_{\\text{t}})
```

# Fields
$(DocStringExtensions.FIELDS)
"""
struct VirtualTemperature{FT} <: AbstractReferenceState{FT}
    "Virtual surface at the temperature"
    T_virt_surf::FT
    "Temperature drop from surface to top of atmosphere (``ΔT = T_surface - T_TOA``)"
    ΔT::FT
    "Height scale over which temperature drops"
    H_t::FT
end

function (ref_state::VirtualTemperature)(
    param_set::APS,
    z::FT,
) where {FT <: AbstractFloat}
    T_virt_surf = ref_state.T_virt_surf
    ΔT = ref_state.ΔT
    H_t = ref_state.H_t

    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)
    _MSLP::FT = MSLP(param_set)

    # Temperature
    T = T_virt_surf - ΔT*tanh(z/H_t)

    # Pressure
    ΔT′ = ΔT/T_virt_surf
    H_sfc = _R_d*T_virt_surf/_grav
    num = z + H_t*ΔT′*log(cosh(z/H_t) - atanh(ΔT′))
    denom = H_sfc*(1 - ΔT′^2)
    exp_factor = num/denom
    p = _MSLP * exp(-exp_factor)

    # Density
    ρ = p / (_R_d * T)
    return T, p, ρ
end

"""
    tested_profiles(param_set, n::Int, ::Type{FT})

A range of input arguments to thermodynamic state constructors
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `q_pt` phase partition
 - `T` air temperature
 - `θ_liq_ice` liquid-ice potential temperature
that are tested for convergence in saturation adjustment.

Note that the output vectors are of size ``n*n_RH``, and they
should span the input arguments to all of the constructors.
"""
function tested_profiles(
    param_set::APS,
    n::Int,
    ::Type{FT}
    ) where {FT}
    test_profile = VirtualTemperature{FT}(290, 60, 8*10^3)

    # TODO: Refactor this (might be able to use broadcast/permutedims):
    z_range = range(FT(0), stop = FT(2.5e4), length = n)
    n_RH = 40
    relative_humidity = range(FT(0), stop = FT(1.2), length=n_RH)
    T = zeros(FT, n, n_RH)
    p = zeros(FT, n, n_RH)
    ρ = zeros(FT, n, n_RH)
    RH = zeros(FT, n, n_RH)
    z_all = zeros(FT, n, n_RH)
    for i in eachindex(z_range)
        for j in eachindex(relative_humidity)
            args = test_profile(param_set, z_range[i])
            k = CartesianIndex(i,j)
            z_all[k] = z_range[i]
            T[k] = args[1]
            p[k] = args[2]
            ρ[k] = args[3]
            RH[k] = relative_humidity[j]
        end
    end
    T = reshape(T, n*n_RH)
    p = reshape(p, n*n_RH)
    ρ = reshape(ρ, n*n_RH)
    RH = reshape(RH, n*n_RH)

    # Total specific humidity
    q_tot = RH .* q_vap_saturation.(Ref(param_set), T, ρ)

    # Compute additional variables:
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot)
    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)

    # Sort by altitude (for visualization):
    # TODO: Refactor, by avoiding phase partition copy, once
    # https://github.com/JuliaLang/julia/pull/33515 merges
    q_liq = getproperty.(q_pt,:liq)
    q_ice = getproperty.(q_pt,:ice)
    args = [z_all, e_int, ρ, q_tot, q_liq, q_ice, T, p, θ_liq_ice]
    args = collect(zip(args...))
    sort!(args)
    z_all     = getindex.(args, 1)
    e_int     = getindex.(args, 2)
    ρ         = getindex.(args, 3)
    q_tot     = getindex.(args, 4)
    q_liq     = getindex.(args, 5)
    q_ice     = getindex.(args, 6)
    T         = getindex.(args, 7)
    p         = getindex.(args, 8)
    θ_liq_ice = getindex.(args, 9)
    q_pt = PhasePartition.(q_tot, q_liq, q_ice)
    args = [z_all, e_int, ρ, q_tot, q_pt, T, p, θ_liq_ice]
    return args
end
